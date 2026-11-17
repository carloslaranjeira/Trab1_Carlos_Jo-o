#!/usr/bin/env python3
"""SAVIBoxes - Trabalho 1 de SAVI. SCRIPT 2 de 2: AVALIADOR (n.º de caixas -> caixas -> result.json).

=====================================================================================================
COMO CORRER
=====================================================================================================
    (a partir da pasta principal do trabalho, DEPOIS do script 1: python savi/detecao_caixas.py --scene ...)
    python savi/avaliador_caixas.py --detection test/cena_1/detecao --num_boxes 2
    python savi/avaliador_caixas.py --detection test/cena_1/detecao        # pergunta o n.º de caixas

Lê a pasta escrita pelo script 1 (detecao_caixas.py): a nuvem de pontos SÓ DAS CAIXAS (caixas.ply) e
o detecao.json (câmaras e mesa). Com o número de caixas indicado, separa as caixas, ajusta-as por
otimização e escreve o result.json (por omissão, na pasta da cena) com a pose de cada câmara e a
pose e as dimensões de cada caixa, tudo no referencial da câmara 000.

Opções extra (não mudam o result.json):
    --evaluate     compara com o ground_truth.json (se a cena o tiver) e mostra tabelas de erros
    --visualize    abre janelas 3D (segmentação, evolução da otimização, caixas finais)
    --figures      grava as figuras top_view.png e optimization.png ao lado do result.json

Avaliação em várias cenas (corre os 2 scripts em cada cena; tabelas em results_offline/results.md):
    python savi/avaliador_caixas.py --evaluate_scenes scenes_offline/level_* scenes_offline/stack_*
    python savi/avaliador_caixas.py --evaluate_scenes scenes_offline/level_* --gt_registration

=====================================================================================================
ORGANIZAÇÃO DESTE FICHEIRO (pela ordem do enunciado)
=====================================================================================================
    3. PARTE 2 - Segmentação  separar as caixas: clusters euclidianos (DBSCAN), n.º de caixas por
                              cluster e cor
    4. PARTE 3 - Ajuste   estimativa inicial + otimização com scipy.optimize.least_squares
    5. Método base        bounding box orientada (só para comparar com a otimização)
    6. Avaliação          erros de uma cena em relação ao ground truth (câmaras, segmentação, caixas, IoU 3D)
    7. Visualização       janelas 3D (Open3D) e figuras PNG (matplotlib)
    8. Várias cenas       corre várias cenas e escreve as tabelas de erros (--evaluate_scenes)
    9. Programa principal ler a deteção do script 1, run() e main()
(As secções 0 a 2 - geometria, leitura da cena e registo - estão no script 1, detecao_caixas.py.
 As convenções - T_A_B, referencial da mesa, caixa = [x, y, yaw, length, width, height, z] - também.)
"""

import argparse
import glob
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2                                   # OpenCV: k-means, retângulos
import matplotlib
matplotlib.use("Agg")                        # o matplotlib só grava ficheiros PNG (não abre janelas)
import matplotlib.pyplot as plt              # figuras (secção 7)
import numpy as np
import open3d as o3d                         # nuvens de pontos: DBSCAN, bounding boxes, janelas 3D
from scipy.optimize import least_squares, linear_sum_assignment   # otimização (Parte 3) / emparelhar caixas (avaliação)
from scipy.spatial import cKDTree            # procura rápida do vizinho mais próximo

# O script 1 está na mesma pasta: junta-a aos sítios onde o Python procura módulos
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detecao_caixas as det                 # noqa: E402 (script 1: deteção)
from detecao_caixas import (  # noqa: E402 (funções do script 1 usadas aqui)
    CLUSTER_EPS, Table, aggregate, box_T_table, box_corners_2d, box_lines_3d, box_signed_distance,
    boxes_to_camera0, load_scene, rotation_angle_deg, transform_points, wrap_angle, write_result, yaw_of,
)


# #####################################################################################################
# 3. PARTE 2 - SEGMENTAÇÃO (2.ª metade): SEPARAR AS CAIXAS
# #####################################################################################################
# Sobre a nuvem SÓ DAS CAIXAS gravada pelo script 1 (referencial da mesa; a mesa e o fundo já saíram):
#   3. separar os objetos com clustering euclidiano (DBSCAN) e decidir quantas caixas tem cada cluster
#      (é aqui que entra o número de caixas);
#   4. dentro de um cluster com várias caixas, tentar separá-las pela cor (Nível 4).
# (Os passos 1 e 2 - mesa e pontos das caixas - estão no script 1.)

@dataclass
class Segmentation:
    """Resultado da segmentação, tudo no referencial da MESA."""
    points: np.ndarray        # M x 3: pontos dos objetos (filtrados e reduzidos)
    colors: np.ndarray        # M x 3: cor RGB de cada ponto, em [0, 1]
    clusters: np.ndarray      # M: índice do cluster euclidiano de cada ponto
    boxes_per_cluster: dict   # cluster -> quantas caixas se pensa estarem nesse cluster
    color_groups: dict        # cluster -> rótulo de cor de cada ponto (ou None se a cor não ajuda)

# ----------------------------------------------------------------------------------------------------
# 3.3 Clustering euclidiano e número de caixas por cluster
# ----------------------------------------------------------------------------------------------------
def euclidean_clusters(points, eps=CLUSTER_EPS, min_points=10, min_cluster_size=150):
    """DBSCAN: junta no mesmo cluster os pontos ligados por cadeias de vizinhos a menos de eps metros.

    Clusters com menos de 'min_cluster_size' pontos são ruído (rótulo -1).
    Saída: rótulo de cada ponto; os clusters ficam numerados do maior (0) para o mais pequeno.
    """
    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
                      .cluster_dbscan(eps=eps, min_points=min_points))
    result = np.full(len(points), -1)
    sizes = [(np.sum(labels == k), k) for k in range(labels.max() + 1)]
    for new_label, (size, k) in enumerate(sorted((s for s in sizes if s[0] >= min_cluster_size), reverse=True)):
        result[labels == k] = new_label
    return result


def footprint_area(points_xy):
    """Área ocupada por um conjunto de pontos visto de cima: área do retângulo de área mínima."""
    (_, _), (a, b), _ = cv2.minAreaRect(points_xy.astype(np.float32))
    return a * b


def large_components(points, eps=0.012, fraction=0.1, min_size=50):
    """Índices das componentes ligadas (DBSCAN) com pelo menos 10% dos pontos da maior (e 50 pontos)."""
    if len(points) < 20:
        return np.arange(len(points))
    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).cluster_dbscan(eps, 5))
    if labels.max() < 0:
        return np.arange(len(points))
    sizes = np.bincount(labels[labels >= 0])
    keep = np.nonzero(sizes >= max(fraction * sizes.max(), min_size))[0]
    return np.nonzero(np.isin(labels, keep))[0]


def components_with(points, n_seed, eps=0.012, min_seed=10, min_size=50):
    """Índices das componentes ligadas (DBSCAN) que pertencem à caixa.

    Os primeiros n_seed pontos são os de confiança (o topo de uma caixa). Fica uma componente que tem
    pelo menos min_seed desses pontos, ou que é grande (min_size pontos): um pedaço de parede pode
    não tocar no topo visível (a caixa de cima tapa o topo quase todo). Saem os grupos pequenos de
    pontos soltos.
    """
    if len(points) < 20:
        return np.arange(len(points))
    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).cluster_dbscan(eps, 5))
    seed = labels[:n_seed]
    keep = [k for k in range(labels.max() + 1)
            if np.sum(seed == k) >= min_seed or np.sum(labels == k) >= min_size]
    if not keep:
        return largest_component(points, eps)
    return np.nonzero(np.isin(labels, keep))[0]


def top_surface_points(points, cell=0.02, tolerance=0.006):
    """Índices dos pontos que estão no TOPO de alguma caixa.

    Divide o plano da mesa numa grelha de células de 2 cm e, em cada célula, fica com os pontos que
    estão a menos de 'tolerance' do ponto mais alto da célula. Os pontos das faces laterais ficam de
    fora (estão mais abaixo do que o topo da mesma célula).
    """
    keys = np.floor(points[:, :2] / cell).astype(np.int64)        # célula de cada ponto
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    cell_max = np.full(inverse.max() + 1, -np.inf)
    np.maximum.at(cell_max, inverse, points[:, 2])                # altura máxima de cada célula
    return np.nonzero((points[:, 2] > cell_max[inverse] - tolerance) & (points[:, 2] > 0.02))[0]


def count_top_levels(points, min_points=30):
    """Número de topos de caixa (superfícies horizontais separadas) num cluster.

    1. Agrupa os pontos do topo (top_surface_points) com DBSCAN em 3D. Grupos com menos de 30 pontos
       (ruído) não contam;
    2. Conta as ALTURAS diferentes desses grupos, e não os grupos: numa pilha, a caixa de cima tapa
       parte do topo da caixa de baixo, e o que se vê desse topo pode ficar partido em vários pedaços
       (um de cada lado da caixa de cima). Pedaços a menos de 1.5 cm de altura uns dos outros são
       o mesmo topo.
    Serve para as caixas empilhadas: uma pilha de 2 tem 2 topos, a alturas diferentes.
    """
    top = points[top_surface_points(points)]
    if len(top) < min_points:
        return 1
    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(top)).cluster_dbscan(0.012, 5))
    heights = sorted(np.median(top[labels == k, 2]) for k in range(labels.max() + 1)
                     if np.sum(labels == k) >= min_points)
    if not heights:
        return 1
    # Alturas ordenadas: um novo topo começa sempre que a altura salta mais de 1.5 cm
    return 1 + int(np.sum(np.diff(heights) > 0.015))


def allocate_boxes(points, clusters, num_boxes):
    """Decide quantas caixas há em cada cluster (o número total de caixas é dado).

    - Tantos clusters como caixas: uma caixa por cluster (Níveis 1 e 2);
    - Menos clusters do que caixas: algumas caixas tocam-se e formam um só cluster (Níveis 3 e 4,
      e caixas empilhadas). As caixas extra vão primeiro para os clusters com mais TOPOS do que
      caixas (uma pilha tem pouca área vista de cima, mas tem vários topos). As que sobram vão,
      uma a uma, para o cluster com maior área por caixa;
    - Mais clusters do que caixas: ficam os maiores (os outros são ruído ou outros objetos).
    Saída: dicionário cluster -> número de caixas.
    """
    labels = list(range(clusters.max() + 1))[:num_boxes]
    counts = {k: 1 for k in labels}
    areas = {k: footprint_area(points[clusters == k, :2]) for k in labels}
    tops = {k: count_top_levels(points[clusters == k]) for k in labels} if num_boxes > len(labels) else {}
    for _ in range(num_boxes - len(labels)):
        missing = [c for c in labels if tops[c] > counts[c]]
        if missing:
            k = max(missing, key=lambda c: areas[c])
        else:
            k = max(labels, key=lambda c: areas[c] / counts[c])
        counts[k] += 1
    return counts


# ----------------------------------------------------------------------------------------------------
# 3.4 Separação por cor (Nível 4)
# ----------------------------------------------------------------------------------------------------
def chroma_features(colors):
    """Converte cores RGB em cromaticidade normalizada (r, g) = (R, G) / (R + G + B), x 100.

    Cada face da caixa recebe uma iluminação diferente: as faces em sombra são muito mais escuras do
    que o topo. A sombra multiplica R, G e B pelo MESMO fator, e esse fator cancela-se na divisão,
    por isso (r, g) é igual em todas as faces da mesma caixa (a mesma ideia do HSV da Parte 02: separar
    a cor do brilho).
    Nota: o Lab (a*, b*) não serve: nas faces escuras (a*, b*) aproximam-se de zero e, numa cena de
    teste, um azul e um roxo deixaram de ser distinguíveis.
    """
    colors = colors.astype(np.float64)
    total = colors.sum(axis=1, keepdims=True) + 1e-3
    return (colors / total)[:, :2] * 100


def color_split(colors, k, min_separation=4.0):
    """Tenta separar um cluster em k grupos de cor com k-means na cromaticidade.

    Saída: rótulo de cor de cada ponto, ou None se as cores não se distinguem (caixas da mesma cor).
    Critério: distância entre os centros de cor dividida pelo desvio padrão (robusto) dentro dos
    grupos tem de ser >= min_separation, e cada grupo tem de ter pelo menos 5% dos pontos.
    """
    if k < 2:
        return None
    features = chroma_features(colors)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    _, labels, centers = cv2.kmeans(features.astype(np.float32), k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    # Desvio padrão robusto: mediana dos desvios absolutos x 1.4826
    spread = np.median(np.linalg.norm(features - centers[labels], axis=1)) * 1.4826 + 1e-6
    center_dist = min(np.linalg.norm(centers[i] - centers[j]) for i in range(k) for j in range(i + 1, k))
    smallest = np.bincount(labels, minlength=k).min() / len(labels)
    if center_dist / spread < min_separation or smallest < 0.05:
        return None
    return labels


def segment(points, colors, num_boxes, use_color=True, verbose=True):
    """Segmentação das caixas: clusters -> caixas por cluster -> cor.

    Entrada: os pontos das caixas (a nuvem gravada pelo script 1, no referencial da mesa, já sem a
    mesa e o fundo), as suas cores e o número de caixas.
    Saída: objeto Segmentation (tudo no referencial da mesa).
    """
    clusters = euclidean_clusters(points)
    counts = allocate_boxes(points, clusters, num_boxes)

    # Pontos de clusters que não foram usados (ruído) ficam de fora
    keep = np.isin(clusters, list(counts.keys()))
    points, colors, clusters = points[keep], colors[keep], clusters[keep]

    # Em cada cluster com várias caixas, tenta separar as caixas pela cor. Primeiro em n cores; se
    # não der (há caixas com a mesma cor), em n-1, ..., 2 cores: por exemplo, duas caixas verdes e
    # uma rosa por cima dão 2 grupos de cor, e o grupo verde é depois dividido pela geometria.
    def partial_color_split(cluster_colors, n):
        for k in range(n, 1, -1):
            labels = color_split(cluster_colors, k)
            if labels is not None:
                return labels
        return None

    color_groups = {k: partial_color_split(colors[clusters == k], n) if (use_color and n > 1) else None
                    for k, n in counts.items()}
    if verbose:
        print(f"  {len(points)} pontos de objetos, {len(counts)} clusters euclidianos; "
              f"caixas por cluster: {counts}; "
              f"clusters separados por cor: {[k for k, g in color_groups.items() if g is not None]}")
    return Segmentation(points, colors, clusters, counts, color_groups)


# #####################################################################################################
# 4. PARTE 3 - AJUSTE DAS CAIXAS POR OTIMIZAÇÃO
# #####################################################################################################
# Função objetivo (um vetor de resíduos que o scipy.optimize.least_squares minimiza, Parte 05/06):
#   1. Dados -> modelo: para cada ponto, a distância à superfície da caixa que melhor o explica.
#      A associação ponto -> caixa é recalculada em CADA avaliação (argmin), sem ciclo exterior;
#   2. Modelo -> dados (cobertura): pontos amostrados nas faces visíveis de cada caixa têm de ter
#      dados por perto. Impede que uma caixa cresça para espaço vazio sem custo;
#   3. Sobreposição: duas caixas não podem ocupar o mesmo espaço;
#   3b. Apoio: uma caixa empilhada tem de assentar no topo da caixa de baixo;
#   4. (opcional) Cor: cada caixa tem uma cor e um ponto escolhe a caixa que melhor explica a sua
#      posição E a sua cor (Nível 4).
# A perda robusta 'soft_l1' (Parte 06, Ex. 5) faz com que uns poucos pontos errados (flying pixels,
# cores trocadas junto às fronteiras) não estraguem o resultado.

NUM_PARAMS = 7            # x, y, yaw, length, width, height, z
FIT_VOXEL = 0.006         # m: os pontos são reduzidos a este voxel antes da otimização
MAX_FIT_POINTS = 12000    # e limitados a este número (velocidade)
GEOMETRIC_SIGMA = 0.003   # m: ruído geométrico típico, usado para pesar a cor
LOSS_SCALE = 0.004        # m: f_scale da perda robusta (resíduos >> 4 mm contam pouco)
HIDDEN_MARGIN = 0.02      # m: faces a menos de 2 cm de outra caixa são consideradas escondidas
ON_TABLE = 0.03           # m: uma caixa com a base a menos de 3 cm da mesa assenta na mesa (z = 0)
NORMAL_PENALTY = 0.02     # m: custo extra de explicar um ponto com uma face de orientação errada


# ----------------------------------------------------------------------------------------------------
# 4.1 Estimativa inicial (Parte 06, Ex. 6: a otimização precisa de começar perto da solução)
# ----------------------------------------------------------------------------------------------------
def rect_params(points_xy):
    """Retângulo de área mínima de pontos 2D (cv2.minAreaRect) -> [x, y, yaw, length, width]."""
    (cx, cy), (a, b), angle = cv2.minAreaRect(points_xy.astype(np.float32))
    return np.array([cx, cy, np.radians(angle), max(a, 0.02), max(b, 0.02)])


def largest_component(points, eps=0.012):
    """Índices da maior componente ligada (DBSCAN) de um conjunto de pontos (retira pontos soltos)."""
    if len(points) < 20:
        return np.arange(len(points))
    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).cluster_dbscan(eps, 5))
    if labels.max() < 0:
        return np.arange(len(points))
    return np.nonzero(labels == np.bincount(labels[labels >= 0]).argmax())[0]


def split_kmeans(group):
    """Divide um grupo de pontos do topo em dois, com k-means no plano (x, y)."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    _, lab, _ = cv2.kmeans(group[:, :2].astype(np.float32), 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    return group[lab == 0], group[lab == 1]


def split_equal(group, k):
    """Divide um topo em k partes IGUAIS, numa grelha de r x s células (r * s = k).

    Caixas iguais da mesma cor encostadas formam um só topo, sem juntas nem degraus: não há nenhuma
    informação sobre onde acaba cada caixa. A hipótese mais simples é que são do mesmo tamanho.
      - 3 caixas em fila: 3 x 1 -> 1/3 + 1/3 + 1/3 (dividir ao meio duas vezes dava 1/2 + 1/4 + 1/4);
      - 4 caixas em quadrado: 2 x 2 (em fila, 4 x 1, davam 4 tiras compridas: teste 7, cena 5).
    Escolhe-se a grelha cujas células (caixas) são MENOS alongadas: as caixas reais não costumam ser
    tiras muito finas.
    """
    _, (a, b), angle = cv2.minAreaRect(group[:, :2].astype(np.float32))
    angle = np.radians(angle) + (0.0 if a >= b else np.pi / 2)   # direção do lado maior
    long_side, short_side = max(a, b), min(a, b)
    u = group[:, :2] @ [np.cos(angle), np.sin(angle)]             # ao longo do lado maior
    v = group[:, :2] @ [-np.sin(angle), np.cos(angle)]            # ao longo do lado menor

    grids = [(r, k // r) for r in range(1, k + 1) if k % r == 0]  # (divisões em u, divisões em v)
    elongation = [max(long_side / r, short_side / s) / max(min(long_side / r, short_side / s), 1e-3)
                  for r, s in grids]
    r, s = grids[int(np.argmin(elongation))]

    iu = np.clip(np.digitize(u, np.linspace(u.min(), u.max(), r + 1)[1:-1]), 0, r - 1)
    iv = np.clip(np.digitize(v, np.linspace(v.min(), v.max(), s + 1)[1:-1]), 0, s - 1)
    cell = iu * s + iv
    return [group[cell == i] for i in range(k) if np.any(cell == i)]


def split_profile(group, bin_size=0.005, min_piece=0.04, min_step=0.025):
    """Divide um grupo de pontos do topo em dois, no 'degrau' do contorno.

    Caixas da mesma altura encostadas formam um topo plano único. Se tiverem larguras diferentes (ou
    estiverem desalinhadas), o contorno desse topo tem um degrau onde acaba uma caixa e começa a outra:
      1. eixo principal do topo (PCA em x, y): u ao longo da fila de caixas, v na perpendicular;
      2. fatias de 5 mm ao longo de u; em cada fatia, os limites do topo em v (percentis 10 e 90);
      3. o degrau é onde estes limites mais mudam (compara a fatia 1 cm antes com a 1 cm depois);
      4. corta-se aí, desde que fiquem pelo menos 4 cm de cada lado.
    Saída: os dois pedaços, ou None se não houver nenhum degrau claro (> 2.5 cm, somando os dois lados).
    (Com 1 cm, o ruído das paredes criava 'degraus' falsos em caixas da mesma largura, e a otimização
    partia daí e ficava num mínimo local: teste 6, cena 13.)
    """
    xy = group[:, :2]
    center = xy.mean(axis=0)
    _, _, Vt = np.linalg.svd(xy - center, full_matrices=False)
    u, v = (xy - center) @ Vt[0], (xy - center) @ Vt[1]

    edges = np.arange(u.min(), u.max() + bin_size, bin_size)
    which = np.digitize(u, edges) - 1                         # fatia de cada ponto
    lo = np.full(len(edges), np.nan)
    hi = np.full(len(edges), np.nan)
    for b in range(len(edges)):
        vb = v[which == b]
        if len(vb) >= 5:
            lo[b], hi[b] = np.percentile(vb, 10), np.percentile(vb, 90)

    k = 2                                                     # 2 fatias = 1 cm
    step = np.full(len(edges), -np.inf)
    for b in range(k, len(edges) - k):
        if edges[b] - u.min() < min_piece or u.max() - edges[b] < min_piece:
            continue
        s = abs(hi[b + k] - hi[b - k]) + abs(lo[b + k] - lo[b - k])
        if not np.isnan(s):
            step[b] = s
    best = int(np.argmax(step))
    if step[best] < min_step:
        return None
    cut = edges[best] + bin_size / 2
    return group[u < cut], group[u >= cut]


def split_corner(group, cell=0.01, min_side=0.04, min_fraction=0.12):
    """Divide um topo em forma de L em 3 retângulos, cortando pelo canto interior do L.

    Três caixas iguais (mesma cor e altura) encostadas em L formam um topo plano único, sem nenhuma
    junta visível. O mais natural é que as juntas passem pelo canto interior do L:
      1. Referencial do retângulo mínimo do topo (eixos u, v); grelha de 1 cm com as células ocupadas;
      2. Em cada um dos 4 cantos do retângulo, procura o maior retângulo VAZIO encostado a esse canto
         (o 'buraco' do L). O seu vértice interior é o canto interior do L;
      3. Corta pelas duas retas (u = a e v = b) que passam por esse vértice: ficam 3 pedaços.
    Saída: os 3 pedaços, ou None se o topo não tiver forma de L (buraco com menos de 12% da área ou
    com algum lado menor do que 4 cm).
    """
    (cx, cy), _, angle = cv2.minAreaRect(group[:, :2].astype(np.float32))
    a = np.radians(angle)
    axes = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    uv = (group[:, :2] - [cx, cy]) @ axes.T                     # coordenadas no referencial do retângulo
    lo = uv.min(axis=0)
    ij = np.floor((uv - lo) / cell).astype(int)
    nu, nv = ij.max(axis=0) + 1
    occupied = np.zeros((nu, nv), int)
    np.add.at(occupied, (ij[:, 0], ij[:, 1]), 1)
    occupied = occupied >= 2                                    # célula com pontos (ignora pontos soltos)

    best = None
    for flip_u in (False, True):                                # os 4 cantos = 4 orientações da grelha
        for flip_v in (False, True):
            grid = occupied[::-1] if flip_u else occupied
            grid = grid[:, ::-1] if flip_v else grid
            # retângulo vazio [0, i) x [0, j) encostado ao canto (0, 0): soma acumulada da ocupação
            cum = grid.cumsum(axis=0).cumsum(axis=1)
            for i in range(1, nu):
                for j in range(1, nv):
                    if cum[i - 1, j - 1] == 0 and (best is None or i * j > best[0]):
                        best = (i * j, i, j, flip_u, flip_v)
    if best is None:
        return None
    area, i, j, flip_u, flip_v = best
    if area < min_fraction * nu * nv or min(i, j) * cell < min_side or min(nu - i, nv - j) * cell < min_side:
        return None
    # Vértice interior do L, em coordenadas (u, v)
    cut_u = lo[0] + (nu - i if flip_u else i) * cell
    cut_v = lo[1] + (nv - j if flip_v else j) * cell
    side_u, side_v = uv[:, 0] >= cut_u, uv[:, 1] >= cut_v
    pieces = [group[(side_u == su) & (side_v == sv)] for su in (False, True) for sv in (False, True)]
    pieces = [p for p in pieces if len(p) >= 30]                # o quadrante vazio desaparece
    return pieces if len(pieces) == 3 else None


def inside_rect(points_xy, rect, margin=0.0):
    """True para os pontos 2D dentro do retângulo [x, y, yaw, length, width] alargado de 'margin'."""
    c, s = np.cos(rect[2]), np.sin(rect[2])
    d = points_xy - rect[:2]
    u, v = d @ [c, s], d @ [-s, c]
    return (np.abs(u) < rect[3] / 2 + margin) & (np.abs(v) < rect[4] / 2 + margin)


def split_geometric(points, n, method="kmeans"):
    """Estimativa inicial de n caixas da MESMA COR que se tocam (um só cluster).

    1. Fica só com os pontos dos topos das caixas e agrupa-os com DBSCAN em 3D;
    2. Junta os grupos que estão à mesma altura (menos de 1.5 cm): é o mesmo NÍVEL. Numa pilha, a
       caixa de cima tapa o meio do topo da caixa de baixo, e o que se vê dele fica partido em vários
       pedaços à volta da caixa de cima; esses pedaços são todos da mesma caixa;
    3. Se houver menos grupos do que caixas, as caixas que faltam vão para os grupos com maior área
       por caixa, e cada grupo com c caixas é dividido em c:
         - c = 3 e forma de L: corta pelo canto interior do L (split_corner);
         - c = 2, method="kmeans":  k-means no plano;
         - c = 2, method="profile": no degrau do contorno (split_profile); sem degrau, partes iguais;
         - restantes casos: c partes iguais, numa grelha (fila ou 2x2) com as caixas menos alongadas (split_equal).
       A divisão é só uma aproximação: a otimização corrige a fronteira. Mas, se começar muito
       longe, pode ficar presa num mínimo local, por isso run() experimenta os dois métodos;
    4. Retângulo de cada caixa visto de cima. Para um nível que não foi dividido e que tem caixas
       mais altas no cluster, o topo visível pode ser só uma moldura ou um pedaço (o resto está
       tapado pela caixa de cima). Juntam-se então as PAREDES dessa caixa: pontos entre o nível de
       baixo e o seu topo, menos os que estão no contorno das caixas mais altas (as paredes delas).
       Se a caixa de cima está rodada e sai para fora, o que se vê da de baixo fica em vários
       pedaços (cantos, um de cada lado): ficam todos os pedaços grandes, não só o maior.
    Saída: lista de n caixas [x, y, yaw, length, width, topo] (ainda sem z: ver place_on_supports).
    """
    top = points[top_surface_points(points)]
    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(top)).cluster_dbscan(0.012, 5))
    pieces = [top[labels == k] for k in range(labels.max() + 1) if np.sum(labels == k) >= 30]

    # 2. Só se há MAIS pedaços do que caixas: junta pedaços à mesma altura (< 1.5 cm), começando pelo
    #    par com alturas mais parecidas. (Se há tantos pedaços como caixas, cada pedaço é uma caixa:
    #    duas caixas iguais encostadas podem ter uma junta visível entre os topos.)
    while len(pieces) > n:
        h = [np.median(g[:, 2]) for g in pieces]
        pairs = [(abs(h[i] - h[j]), i, j) for i in range(len(pieces)) for j in range(i + 1, len(pieces))]
        diff, i, j = min(pairs)
        if diff >= 0.015:
            break
        pieces[i] = np.vstack([pieces[i], pieces.pop(j)])
    pieces.sort(key=len, reverse=True)
    groups = [(g, True) for g in pieces[:n]] if pieces else [(top, False)]   # (pontos, não dividido?)

    # 3. Quantas caixas há em cada grupo: uma cada; as que faltam vão, uma a uma, para o grupo com
    #    maior área por caixa (três caixas iguais em fila são um só topo com 3 caixas)
    areas = [rect_params(g[:, :2])[3] * rect_params(g[:, :2])[4] for g, _ in groups]
    counts = [1] * len(groups)
    for _ in range(n - len(groups)):
        counts[int(np.argmax(np.array(areas) / counts))] += 1
    divided = []
    for (g, whole_level), c in zip(groups, counts):
        if c == 1:
            divided.append((g, whole_level))
            continue
        split = split_corner(g) if c == 3 else None             # forma de L
        if split is None and c == 2 and method == "profile":
            split = split_profile(g)                            # degrau do contorno
        if split is None:
            split = split_kmeans(g) if c == 2 and method == "kmeans" else split_equal(g, c)
        divided += [(p, False) for p in split]
    groups = divided

    # 4. Retângulos, do grupo mais alto para o mais baixo
    groups.sort(key=lambda item: -np.median(item[0][:, 2]))
    heights = [np.median(g[:, 2]) for g, _ in groups]
    boxes = []
    for i, (g, whole_level) in enumerate(groups):
        rect = rect_params(g[:, :2])
        # Só para um grupo sozinho na sua altura e com caixas mais altas no cluster (numa pilha);
        # se houver outra caixa à mesma altura, as paredes dela misturavam-se com as desta.
        alone = sum(abs(h - heights[i]) < 0.015 for h in heights) == 1
        if whole_level and alone and heights[0] > heights[i] + 0.015:
            below = max([h for h in heights if h < heights[i] - 0.015], default=0.0)
            wall = points[(points[:, 2] > below + 0.01) & (points[:, 2] < heights[i] - 0.01)]
            # Tira as paredes das caixas mais altas: estão no contorno delas (a menos de 1-1.5 cm).
            # Os pontos BEM dentro do retângulo de uma caixa mais alta ficam: são paredes desta caixa,
            # por baixo de uma parte da caixa de cima que sai para fora (caixa de cima rodada).
            for b in boxes:
                wall = wall[~(inside_rect(wall[:, :2], b[:5], margin=0.01) &
                              ~inside_rect(wall[:, :2], b[:5], margin=-0.015))]
            candidates = np.vstack([g, wall])
            # Fica com todas as componentes ligadas que têm pontos do topo desta caixa (e não só a
            # maior): se a caixa de cima está rodada e sai para fora, divide o que se vê da caixa de
            # baixo em pedaços separados (um de cada lado, ou só os cantos), todos da mesma caixa.
            candidates = candidates[components_with(candidates, len(g))]
            rect = rect_params(candidates[:, :2])
        boxes.append(np.append(rect, heights[i]))
    return boxes


def place_on_supports(boxes, min_cover=0.5, min_step=0.02):
    """Decide onde assenta cada caixa: na mesa ou em cima de outra (caixas empilhadas).

    Entrada: caixas [x, y, yaw, length, width, topo].
    Uma caixa B está por cima de uma caixa A se o topo de A está pelo menos 2 cm abaixo do topo de B
    e, vistos de cima, os retângulos se sobrepõem em pelo menos metade da área do mais pequeno
    (caixas encostadas lado a lado quase não se sobrepõem vistas de cima). A base de B fica no topo
    da caixa mais alta que está por baixo dela; sem nenhuma por baixo, fica na mesa (z = 0).
    Saída: caixas completas [x, y, yaw, length, width, height, z], com height = topo - z.
    """
    out = []
    for j, b in enumerate(boxes):
        z = 0.0
        for k, a in enumerate(boxes):
            if k == j or a[5] > b[5] - min_step:              # A tem de ser mais baixa do que B
                continue
            smaller = min(a[3] * a[4], b[3] * b[4])
            if overlap_area(a, b) > min_cover * smaller:
                z = max(z, a[5])                              # assenta no topo de A
        out.append(np.array([b[0], b[1], b[2], b[3], b[4], max(b[5] - z, 0.02), z]))
    return out


def needs_geometric_split(seg):
    """True se algum cluster tem várias caixas que a cor não separa (Nível 3 ou pilhas da mesma cor)."""
    groups = seg.color_groups
    return any(n > 1 and (groups.get(k) is None or groups[k].max() + 1 < n)
               for k, n in seg.boxes_per_cluster.items())


def initial_boxes(seg, split_method="kmeans"):
    """Estimativa inicial de todas as caixas, a partir da segmentação.

    - Cluster com uma caixa: retângulo mínimo do cluster visto de cima e topo = percentil 98 de z;
    - Cluster com várias caixas de cores distintas: o mesmo, para cada grupo de cor;
    - Cluster com várias caixas da mesma cor: split_geometric (topos + divisão).
    Em todos os casos a base (z) vem de place_on_supports (caixas empilhadas assentam na de baixo).
    Saída: (lista de caixas, cor de cada caixa, rótulo da caixa de cada ponto da segmentação).
    Os rótulos são os 'conjuntos de pontos' de cada caixa, usados pela bounding box e na avaliação.
    """
    params, colors = [], []
    labels = np.full(len(seg.points), -1)
    for k, n in seg.boxes_per_cluster.items():
        idx = np.nonzero(seg.clusters == k)[0]                 # pontos deste cluster
        pts, cols = seg.points[idx], seg.colors[idx]
        groups = seg.color_groups.get(k)
        first = len(params)                                     # índice da 1.ª caixa deste cluster
        if n == 1:
            box = np.append(rect_params(pts[:, :2]), np.percentile(pts[:, 2], 98))
            params.append(place_on_supports([box])[0])
            colors.append(np.median(chroma_features(cols), axis=0))
            labels[idx] = first
        elif groups is not None:
            # Quantas caixas há em cada grupo de cor (menos cores do que caixas: há caixas da mesma
            # cor); a mesma regra que para os clusters (topos a alturas diferentes, depois área)
            per_color = allocate_boxes(pts, groups, n)
            tops = []
            for g, c in per_color.items():
                sel = np.nonzero(groups == g)[0]
                # Retira pontos soltos com a cor errada. Com uma caixa desta cor fica a maior componente;
                # com várias, ficam as componentes grandes: numa torre com cores alternadas (laranja,
                # rosa, laranja, rosa) as caixas da mesma cor não se tocam (teste 11, cena 6).
                sel = sel[largest_component(pts[sel]) if c == 1 else large_components(pts[sel])]
                color = np.median(chroma_features(cols[sel]), axis=0)
                if c == 1:
                    labels[idx[sel]] = first + len(tops)
                    tops.append(np.append(rect_params(pts[sel, :2]), np.percentile(pts[sel, 2], 98)))
                    colors.append(color)
                else:                                           # várias caixas desta cor: geometria
                    for box in split_geometric(pts[sel], c, split_method):
                        tops.append(box)
                        colors.append(color)
            params += place_on_supports(tops)
        else:
            for p in place_on_supports(split_geometric(pts, n, split_method)):
                params.append(p)
                colors.append(np.median(chroma_features(cols), axis=0))
        # Pontos ainda sem caixa: ficam com a caixa (deste cluster) cuja superfície está mais perto
        free = labels[idx] < 0
        if free.any():
            D = distances_to_boxes(pts[free], params[first:first + n])
            labels[idx[free]] = first + np.argmin(D, axis=1)
    return params, colors, labels


# ----------------------------------------------------------------------------------------------------
# 4.2 Modelo da caixa
# ----------------------------------------------------------------------------------------------------
def to_box_frame(points, p):
    """Escreve pontos do referencial da mesa no referencial da caixa p (origem no centro da caixa)."""
    x, y, yaw, _, _, h, z = p
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = points[:, 0] - x, points[:, 1] - y
    # Rotação inversa do yaw (R^T) e translação para o centro da caixa (a z + h/2 da mesa)
    return np.stack([c * dx + s * dy, -s * dx + c * dy, points[:, 2] - (z + h / 2)], axis=1)


def distances_to_boxes(points, params, normals=None):
    """Matriz M x B com a distância de cada ponto à superfície de cada caixa.

    Sem normais: distância à superfície (valor absoluto da signed distance function).

    Com normais: distância à FACE mais próxima cuja orientação é compatível com a normal do ponto.
    Porquê? Quando duas caixas se tocam, os pontos do topo da caixa mais baixa junto à caixa mais
    alta estão a poucos mm da face LATERAL da alta. Só com distâncias, a caixa baixa podia encolher
    alguns mm sem o erro aumentar (esses pontos passavam a ser 'explicados' pela face lateral).
    Exigindo que um ponto com normal vertical seja explicado por uma face horizontal (e vice-versa),
    a ambiguidade desaparece (é a mesma ideia de filtrar pelas normais da Parte 07, Ex. 6-8).
    Faces de orientação errada não são proibidas, só custam mais NORMAL_PENALTY (as normais nas
    arestas não são fiáveis). A face de baixo nunca conta: assenta na mesa ou noutra caixa.
    """
    D = np.empty((len(points), len(params)))
    for j, p in enumerate(params):
        half = np.array([p[3], p[4], p[5]]) / 2                   # meias-dimensões
        local = to_box_frame(points, p)
        if normals is None:
            D[:, j] = np.abs(box_signed_distance(local, half))
            continue
        # Normais no referencial da caixa (só a rotação do yaw); o sinal não interessa (abs)
        c, s = np.cos(p[2]), np.sin(p[2])
        n_local = np.abs(np.stack([c * normals[:, 0] + s * normals[:, 1],
                                   -s * normals[:, 0] + c * normals[:, 1], normals[:, 2]], axis=1))
        best = np.full(len(points), np.inf)
        for axis in range(3):                    # face perpendicular ao eixo x, y ou z da caixa
            for sign in (-1.0, 1.0):             # face do lado - ou do lado +
                if axis == 2 and sign < 0:
                    continue                     # a face de baixo assenta na mesa ou noutra caixa
                # Projeção do ponto no retângulo da face: fixa a coordenada do eixo e limita as outras
                proj = np.clip(local, -half, half)
                proj[:, axis] = sign * half[axis]
                d = np.linalg.norm(local - proj, axis=1)
                d += NORMAL_PENALTY * (n_local[:, axis] < 0.7)     # orientação incompatível
                best = np.minimum(best, d)
        D[:, j] = best
    return D


def unit_surface_samples(n=7):
    """Amostras (e a normal de cada uma) na superfície de uma caixa unitária ([-1, 1] em cada eixo).

    n x n amostras no topo e em cada uma das 4 faces laterais. Não há amostras na face de baixo (nunca
    é vista) nem na parte de baixo das faces laterais (os pontos junto à mesa são cortados na Parte 2).
    São multiplicadas pelas meias-dimensões de cada caixa em surface_samples.
    """
    t = np.linspace(-1, 1, n)
    a, b = np.meshgrid(t, t)
    a, b = a.ravel(), b.ravel()
    one = np.ones_like(a)
    faces = [(np.stack([a, b, one], 1), (0, 0, 1)),                                        # topo
             (np.stack([one, a, b], 1), (1, 0, 0)), (np.stack([-one, a, b], 1), (-1, 0, 0)),   # +x e -x
             (np.stack([a, one, b], 1), (0, 1, 0)), (np.stack([a, -one, b], 1), (0, -1, 0))]   # +y e -y
    samples = np.concatenate([f for f, _ in faces])
    normals = np.concatenate([np.tile(nrm, (len(f), 1)) for f, nrm in faces]).astype(float)
    keep = samples[:, 2] > -0.75
    return samples[keep], normals[keep]


UNIT_SAMPLES, UNIT_NORMALS = unit_surface_samples()     # calculadas uma só vez


def surface_samples(p):
    """Amostras na superfície da caixa p e a normal da face de cada uma (referencial da mesa)."""
    x, y, yaw, l, w, h, z = p
    local = UNIT_SAMPLES * np.array([l, w, h]) / 2
    c, s = np.cos(yaw), np.sin(yaw)
    points = np.stack([x + c * local[:, 0] - s * local[:, 1],
                       y + s * local[:, 0] + c * local[:, 1],
                       local[:, 2] + z + h / 2], axis=1)
    normals = np.stack([c * UNIT_NORMALS[:, 0] - s * UNIT_NORMALS[:, 1],
                        s * UNIT_NORMALS[:, 0] + c * UNIT_NORMALS[:, 1], UNIT_NORMALS[:, 2]], axis=1)
    return points, normals


def facing_some_camera(samples, normals, camera_centers, min_cos=0.2):
    """True para as amostras cuja face está virada para pelo menos uma câmara.

    Uma face só pode ser vista de uma câmara se a normal apontar para o lado da câmara:
    cos(ângulo entre a normal e a direção amostra -> câmara) > min_cos (~78°; faces quase de perfil
    também não dão pontos). Com poucas câmaras há faces que nenhuma câmara vê: não têm pontos, e
    isso não quer dizer que a caixa seja mais pequena. (Não considera oclusões por outras caixas.)
    """
    if camera_centers is None:
        return np.ones(len(samples), bool)
    to_cam = camera_centers[None, :, :] - samples[:, None, :]             # S x C x 3
    to_cam /= np.linalg.norm(to_cam, axis=2, keepdims=True)
    cos = np.einsum("sck,sk->sc", to_cam, normals)                        # produto interno
    return (cos > min_cos).any(axis=1)


def height_overlap(p, q):
    """Sobreposição (m) dos intervalos de altura [z, z + height] de duas caixas (0 se não se cruzam)."""
    return max(0.0, min(p[6] + p[5], q[6] + q[5]) - max(p[6], q[6]))


def overlap_area(p, q):
    """Área da interseção dos retângulos de duas caixas vistas de cima (cv2.rotatedRectangleIntersection)."""
    r1 = ((p[0], p[1]), (p[3], p[4]), np.degrees(p[2]))
    r2 = ((q[0], q[1]), (q[3], q[4]), np.degrees(q[2]))
    status, region = cv2.rotatedRectangleIntersection(r1, r2)
    if status == cv2.INTERSECT_NONE or region is None or len(region) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(region.reshape(-1, 2).astype(np.float32))))


def normalize_box(p):
    """Põe a caixa numa forma canónica: length >= width e yaw em [-90°, 90°).

    Trocar length com width e rodar 90° (ou rodar 180°) dá exatamente a mesma caixa.
    """
    x, y, yaw, l, w, h, z = p
    if w > l:
        l, w, yaw = w, l, yaw + np.pi / 2
    yaw = wrap_angle(2 * yaw) / 2                 # módulo 180°
    return np.array([x, y, yaw, l, w, h, z])


# ----------------------------------------------------------------------------------------------------
# 4.3 Função objetivo
# ----------------------------------------------------------------------------------------------------
@dataclass
class FitProblem:
    """Tudo o que a função objetivo precisa. O método residuals() é chamado pelo least_squares."""
    points: np.ndarray                 # M x 3, pontos dos objetos no referencial da mesa
    tree: cKDTree                      # KD-tree dos pontos (para o termo modelo -> dados)
    num_boxes: int
    normals: np.ndarray = None         # M x 3, normal estimada de cada ponto
    camera_centers: np.ndarray = None  # C x 3, posição das câmaras no referencial da mesa
    chroma: np.ndarray = None          # M x 2, cor (cromaticidade) de cada ponto, se a cor for usada
    box_chroma: np.ndarray = None      # B x 2, cor de cada caixa
    color_weight: float = 0.0          # lambda: converte diferenças de cor em 'metros equivalentes'
    stacked: tuple = ()                # índices das caixas empilhadas (z livre)
    coverage_weight: float = 1.0
    overlap_weight: float = 1.0
    history: list = field(default_factory=list)   # (parâmetros, custo) de cada avaliação, para as figuras

    def residuals(self, x):
        """Vetor de resíduos para os parâmetros x (todas as caixas seguidas, 7 números cada)."""
        params = x.reshape(self.num_boxes, NUM_PARAMS)
        M = len(self.points)

        # 1. Dados -> modelo, com a associação ponto -> caixa recalculada AQUI (em cada avaliação)
        D = distances_to_boxes(self.points, params, self.normals)
        if self.chroma is not None:
            # Cor: distância de cor de cada ponto à cor de cada caixa, pesada por lambda
            C = np.linalg.norm(self.chroma[:, None, :] - self.box_chroma[None], axis=2) * self.color_weight
            assoc = np.argmin(D ** 2 + C ** 2, axis=1)         # caixa que melhor explica posição + cor
            rows = np.arange(M)
            res_points = np.concatenate([D[rows, assoc], C[rows, assoc]])
        else:
            assoc = np.argmin(D, axis=1)                        # caixa mais próxima do ponto
            res_points = D[np.arange(M), assoc]

        # 2. Modelo -> dados: cada amostra da superfície tem de ter um ponto medido por perto.
        #    Não contam: faces que nenhuma câmara vê, e amostras dentro ou a menos de HIDDEN_MARGIN de
        #    outra caixa (faces de contacto, invisíveis). Uma margem pequena (ex.: 5 mm) criava uma
        #    barreira: a face só deixava de ser penalizada quando já estava encostada à outra caixa e
        #    o otimizador ficava parado a ~1 cm dela.
        res_cover = []
        # Fator de escala (adaptação do sqrt(n) da Parte 06, Ex. 3): há num_boxes * S resíduos de
        # cobertura e M resíduos dos pontos. Multiplicar cada um por sqrt(M / (num_boxes * S)) faz com
        # que o conjunto da cobertura pese tanto como o conjunto dos pontos (a soma dos quadrados fica
        # com o mesmo número 'efetivo' de termos).
        scale = np.sqrt(M / (self.num_boxes * len(UNIT_SAMPLES)))
        for j, p in enumerate(params):
            samples, sample_normals = surface_samples(p)
            hidden = ~facing_some_camera(samples, sample_normals, self.camera_centers)
            for k, q in enumerate(params):
                if k != j:
                    hidden |= box_signed_distance(to_box_frame(samples, q), np.array(q[3:6]) / 2) < HIDDEN_MARGIN
            dist, _ = self.tree.query(samples, k=1)                 # ponto medido mais próximo
            dist[hidden] = 0.0
            res_cover.append(self.coverage_weight * scale * dist)

        # Parte 06, Ex. 3: os resíduos 'globais' da sobreposição e do apoio são POUCOS (um por par de
        # caixas / por caixa empilhada) e os dos pontos são M. Por isso cada um é multiplicado por
        # sqrt(M), exatamente como na aula (erros de brilho/contraste * sqrt(n) ao lado dos n erros
        # de píxel): ao quadrado fica M * erro^2, isto é, cada erro global vale tanto como os M pontos juntos.
        # 3. Sobreposição: raiz da área da interseção vista de cima (unidades de comprimento). Só conta
        #    se as alturas também se cruzarem: o fator cresce de 0 a 1 no primeiro centímetro de
        #    interpenetração, para uma caixa empilhada poder encostar à de baixo sem custo.
        res_overlap = [self.overlap_weight * np.sqrt(M) * np.sqrt(overlap_area(params[i], params[j]))
                       * min(1.0, height_overlap(params[i], params[j]) / 0.01)
                       for i in range(self.num_boxes) for j in range(i + 1, self.num_boxes)]

        # 3b. Apoio: uma caixa empilhada (z livre) tem de assentar no topo da caixa mais alta que está
        #     por baixo dela (sobreposição vista de cima e topo abaixo da base + 3 cm)
        res_support = []
        for j in self.stacked:
            p = params[j]
            below = [q[6] + q[5] for k, q in enumerate(params)
                     if k != j and q[6] + q[5] < p[6] + 0.03 and overlap_area(p, q) > 0]
            support = max(below) if below else 0.0             # sem nada por baixo: a mesa
            res_support.append(self.overlap_weight * np.sqrt(M) * (p[6] - support))

        residuals = np.concatenate([res_points] + res_cover +
                                   [np.array(res_overlap), np.array(res_support)])
        # Guarda o custo (como o soft_l1 o calcula) para as figuras da evolução da otimização
        self.history.append((x.copy(), float(np.sum(np.sqrt(1 + (residuals / LOSS_SCALE) ** 2) - 1))))
        return residuals

    def associations(self, x):
        """Caixa associada a cada ponto, para os parâmetros x (para avaliar a segmentação final)."""
        params = x.reshape(self.num_boxes, NUM_PARAMS)
        D = distances_to_boxes(self.points, params, self.normals)
        if self.chroma is not None:
            C = np.linalg.norm(self.chroma[:, None, :] - self.box_chroma[None], axis=2) * self.color_weight
            return np.argmin(D ** 2 + C ** 2, axis=1)
        return np.argmin(D, axis=1)


def evolution_snapshots(history):
    """Do histórico de avaliações, fica só com as que melhoraram o custo (a 'trajetória')."""
    best, snapshots = np.inf, []
    for x, cost in history:
        if cost < best - 1e-9:
            best = cost
            snapshots.append(x)
    return snapshots


def fit_boxes(seg, init_params, init_colors, use_color=True, use_normals=True, camera_centers=None,
              verbose=True):
    """Ajusta todas as caixas AO MESMO TEMPO com scipy.optimize.least_squares.

    Entradas: segmentação, estimativa inicial (caixas e cores), se usa a cor e as normais, e a
    posição das câmaras no referencial da mesa (para não exigir pontos em faces que ninguém vê).
    1. Reduz os pontos (voxel de 6 mm e no máximo 12000 pontos) para a otimização ser rápida;
    2. Estima a normal de cada ponto;
    3. Peso da cor: lambda = sigma_geométrico / sigma_cor, isto é, cada erro é dividido pelo seu
       desvio padrão, para 1 desvio de cor 'valer' 1 desvio de distância (põe as duas grandezas na
       mesma unidade; não é o sqrt(n) da Parte 06 Ex. 3, que é usado na cobertura, sobreposição e apoio);
    4. O z das caixas assentes na mesa fica fixo em 0 (não é otimizado); só o das empilhadas é livre.
       Se fosse livre em todas, uma caixa na mesa podia 'subir' alguns mm sem custo (a segmentação
       corta os pontos até 8 mm acima da mesa);
    5. least_squares com perda soft_l1 e limites nas dimensões.
    Saída: (caixas finais N x 7, o problema (com o histórico), pontos usados, associações finais).
    """
    # 1. Redução dos pontos
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seg.points))
    pc.colors = o3d.utility.Vector3dVector(seg.colors)
    pc = pc.voxel_down_sample(FIT_VOXEL)
    points, colors = np.asarray(pc.points), np.asarray(pc.colors)
    if len(points) > MAX_FIT_POINTS:
        sel = np.random.default_rng(0).choice(len(points), MAX_FIT_POINTS, replace=False)
        points, colors = points[sel], colors[sel]

    # 2. Normais de cada ponto (PCA dos vizinhos num raio de 1.5 cm); o sinal não interessa
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
    normals = np.asarray(pc.normals)

    n = len(init_params)
    problem = FitProblem(points, cKDTree(points), n, normals=normals if use_normals else None)
    problem.camera_centers = None if camera_centers is None else np.asarray(camera_centers)

    # 3. A cor só entra se a segmentação mostrou que as caixas têm cores distintas
    color_informative = any(g is not None for g in seg.color_groups.values())
    if use_color and color_informative and n > 1:
        problem.chroma = chroma_features(colors)
        problem.box_chroma = np.array(init_colors)
        nearest = np.argmin(np.linalg.norm(problem.chroma[:, None] - problem.box_chroma[None], axis=2), axis=1)
        spread = np.median(np.linalg.norm(problem.chroma - problem.box_chroma[nearest], axis=1)) * 1.4826
        problem.color_weight = GEOMETRIC_SIGMA / max(spread, 1e-3)

    # 4. Parâmetros livres e fixos (z = 0 fixo nas caixas assentes na mesa)
    x_full = np.concatenate(init_params).astype(float)
    problem.stacked = tuple(j for j, p in enumerate(init_params) if p[6] > ON_TABLE)
    free = np.ones(len(x_full), bool)
    for j in range(n):
        if j not in problem.stacked:
            free[j * NUM_PARAMS + 6] = False
            x_full[j * NUM_PARAMS + 6] = 0.0

    # Limites: dimensões entre 2 cm e 1 m, base entre a mesa e 1 m; posição e yaw livres
    lower = np.tile([-np.inf, -np.inf, -np.inf, 0.02, 0.02, 0.02, 0.0], n)[free]
    upper = np.tile([np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, 1.0], n)[free]
    x0 = np.clip(x_full[free], lower + 1e-6, upper - 1e-6)
    # Escala típica de cada parâmetro (ajuda o otimizador): 1 cm nas distâncias, ~3° no yaw
    x_scale = np.tile([0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01], n)[free]

    def residuals_free(x_free):
        """Junta os parâmetros otimizados com os fixos e calcula os resíduos."""
        x = x_full.copy()
        x[free] = x_free
        return problem.residuals(x)

    # 5. Otimização
    result = least_squares(residuals_free, x0, bounds=(lower, upper), loss="soft_l1",
                           f_scale=LOSS_SCALE, x_scale=x_scale, diff_step=1e-4, max_nfev=200)
    x_final = x_full.copy()
    x_final[free] = result.x
    params = np.array([normalize_box(p) for p in x_final.reshape(n, NUM_PARAMS)])
    if verbose:
        print(f"  otimização: {result.nfev} iterações, custo {problem.history[0][1]:.1f} -> "
              f"{min(h[1] for h in problem.history):.1f}, "
              f"cor {'sim' if problem.chroma is not None else 'não'} ({len(points)} pontos)")
    return params, problem, points, problem.associations(x_final)


# #####################################################################################################
# 5. MÉTODO BASE: BOUNDING BOX ORIENTADA (só para comparar com a otimização)
# #####################################################################################################
def bbox_box(points, minimal=True, stacked=False):
    """Bounding box orientada (Parte 08, Ex. 5) de um conjunto de pontos -> [x, y, yaw, l, w, h, z].

    1. Remove outliers (estatístico + maior componente ligada), para tratar o método de forma justa;
    2. Calcula a bounding box com get_minimal_oriented_bounding_box() (menor volume) ou
       get_oriented_bounding_box() (PCA);
    3. O eixo z da caixa tem de ser perpendicular à mesa: dos 3 eixos da bounding box, o mais
       alinhado com o z da mesa passa a ser a altura; os outros dois dão o yaw e as dimensões;
    4. Caixa na mesa: a base é posta em z = 0 e a altura vai até ao topo da bounding box (sem isto
       a caixa perdia os ~8 mm cortados junto à mesa). Caixa empilhada ('stacked', decidido na
       estimativa inicial): a base é o fundo da bounding box. Não basta olhar para o fundo: uma
       caixa da qual só se viu o topo teria o fundo no ar.
    """
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pts = np.asarray(pc.points)
    pts = pts[largest_component(pts)]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    obb = pc.get_minimal_oriented_bounding_box() if minimal else pc.get_oriented_bounding_box()
    R, extent, center = np.asarray(obb.R), np.asarray(obb.extent), np.asarray(obb.center)

    k = int(np.argmax(np.abs(R[2, :])))           # eixo da bounding box mais alinhado com o z da mesa
    others = [i for i in range(3) if i != k]
    axis_x = R[:, others[0]]
    yaw = np.arctan2(axis_x[1], axis_x[0])
    length, width = extent[others[0]], extent[others[1]]
    top = center[2] + extent[k] / 2
    base = center[2] - extent[k] / 2
    z = base if stacked and base > ON_TABLE else 0.0
    return normalize_box(np.array([center[0], center[1], yaw, length, width, top - z, z]))


def bbox_method(seg_points, box_labels, num_boxes, minimal=True, init_params=None):
    """Aplica a bounding box a cada conjunto de pontos (box_labels[i] = caixa do ponto i).

    init_params: estimativa inicial (diz que caixas estão empilhadas: z > 0).
    """
    boxes = []
    for j in range(num_boxes):
        pts = seg_points[box_labels == j]
        if len(pts) < 10:
            continue
        stacked = init_params is not None and init_params[j][6] > ON_TABLE
        boxes.append(bbox_box(pts, minimal, stacked))
    return np.array(boxes)


# #####################################################################################################
# 6. AVALIAÇÃO DE UMA CENA (comparação com o ground truth; só para avaliar, NUNCA para estimar)
# #####################################################################################################
# Usado pela opção --evaluate, pela opção --evaluate_scenes e pelo run_test.py.
#   - Registo: erro de translação (mm) e de rotação (graus) de cada câmara;
#   - Segmentação: n.º de conjuntos encontrados, precisão e recall de cada conjunto de pontos;
#   - Caixas: erro de posição (mm), de yaw (graus), das dimensões (mm), da altura da base (mm) e IoU 3D,
#     tendo em conta as simetrias da caixa (rodar 180° ou trocar comprimento e largura é a mesma caixa).

# --------------------------------------------------------------------------------------
# Registo
# --------------------------------------------------------------------------------------
def camera_errors(T_est_list, ground_truth):
    """Lista de (imagem, erro de translação em mm, erro de rotação em graus)."""
    rows = []
    for T_est, cam in zip(T_est_list, ground_truth["cameras"]):
        T_gt = np.array(cam["T_camera0_camera"])
        t_err = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]) * 1000
        r_err = rotation_angle_deg(T_est[:3, :3].T @ T_gt[:3, :3])   # ângulo de R_est^T R_gt
        rows.append((cam["image"], t_err, r_err))
    return rows


# --------------------------------------------------------------------------------------
# Caixas
# --------------------------------------------------------------------------------------
def gt_params_in_table(ground_truth, T_camera0_table):
    """Caixas verdadeiras no referencial da mesa estimada (só para desenhar nas figuras)."""
    T_table_camera0 = np.linalg.inv(T_camera0_table)
    params = []
    for b in ground_truth["boxes"]:
        T = T_table_camera0 @ np.array(b["T_camera0_box"])
        z_base = T[2, 3] - b["height"] / 2               # altura da base acima da mesa
        params.append([T[0, 3], T[1, 3], yaw_of(T[:3, :3]), b["length"], b["width"], b["height"], z_base])
    return np.array(params)


def print_evaluation(r):
    """Mostra no terminal as tabelas de erros de uma cena (r = resultado de run())."""
    gt = r["ground_truth"]
    print("\nRegisto (erro de cada câmara):")
    print(f"  {'imagem':8s}{'transl. (mm)':>14s}{'rot. (°)':>10s}")
    for name, t, a in camera_errors(r["T_cams"], gt):
        print(f"  {name:8s}{t:14.2f}{a:10.3f}")

    n_sets, n_gt, seg_rows = evaluate_segmentation(
        r["seg"].points, r["box_labels"], r["table"].T_camera0_table, full_cloud_points(r), gt)
    print(f"\nSegmentação: {n_sets} conjuntos para {n_gt} caixas")
    for row in seg_rows:
        print(f"  caixa {row['gt_box']}: precisão {row['precision']:.3f}, recall {row['recall']:.3f}")

    T_ct = r["table"].T_camera0_table
    for name in ("bbox", "opt"):
        rows = evaluate_boxes(boxes_to_camera0(r[f"params_{name}"], T_ct), gt)
        print(f"\nCaixas - {'bounding box' if name == 'bbox' else 'otimização'}:")
        print(f"  {'caixa':6s}{'pos (mm)':>10s}{'yaw (°)':>9s}{'dL (mm)':>9s}{'dW (mm)':>9s}{'dH (mm)':>9s}"
              f"{'dz (mm)':>9s}{'IoU':>7s}")
        for j, e in enumerate(rows):
            print(f"  {j:<6d}{e['pos_mm']:10.1f}{e['yaw_deg']:9.2f}{e['dl_mm']:9.1f}{e['dw_mm']:9.1f}"
                  f"{e['dh_mm']:9.1f}{e['dz_mm']:9.1f}{e['iou']:7.3f}")
        print(f"  IoU médio: {np.mean([e['iou'] for e in rows]):.3f}")


def _box_in_world(T_world_box, length, width, height):
    """Descrição de uma caixa no referencial do mundo (z para cima): centro, yaw e dimensões."""
    return {"center": T_world_box[:3, 3], "yaw": yaw_of(T_world_box[:3, :3]),
            "dims": np.array([length, width, height])}


def iou_3d(a, b):
    """IoU 3D de duas caixas com z para cima.

    Volume da interseção = área da interseção dos retângulos vistos de cima
                           x sobreposição dos intervalos de altura.
    """
    r1 = (tuple(a["center"][:2]), tuple(a["dims"][:2]), np.degrees(a["yaw"]))
    r2 = (tuple(b["center"][:2]), tuple(b["dims"][:2]), np.degrees(b["yaw"]))
    status, region = cv2.rotatedRectangleIntersection(r1, r2)
    area = 0.0
    if status != cv2.INTERSECT_NONE and region is not None and len(region) >= 3:
        area = cv2.contourArea(cv2.convexHull(region.reshape(-1, 2).astype(np.float32)))
    z1 = (a["center"][2] - a["dims"][2] / 2, a["center"][2] + a["dims"][2] / 2)
    z2 = (b["center"][2] - b["dims"][2] / 2, b["center"][2] + b["dims"][2] / 2)
    dz = max(0.0, min(z1[1], z2[1]) - max(z1[0], z2[0]))
    inter = area * dz
    union = np.prod(a["dims"]) + np.prod(b["dims"]) - inter
    return inter / union if union > 0 else 0.0


def box_errors(est, gt):
    """Erros de uma caixa estimada em relação à verdadeira, com as simetrias.

    Versões equivalentes da caixa verdadeira: rodar k*90° em torno de z (k = 0..3), trocando
    length com width quando k é ímpar. Ficamos com a versão de menor erro de yaw.
    """
    pos_err = np.linalg.norm(est["center"] - gt["center"]) * 1000
    best = None
    for k in range(4):
        yaw_err = abs(np.degrees(wrap_angle(est["yaw"] - gt["yaw"] - k * np.pi / 2)))
        dims_gt = gt["dims"][[1, 0, 2]] if k % 2 else gt["dims"]
        dim_err = (est["dims"] - dims_gt) * 1000
        score = yaw_err + np.abs(dim_err).sum() * 0.01      # yaw primeiro; dimensões desempatam (caixas quadradas)
        if best is None or score < best[0]:
            best = (score, yaw_err, dim_err)
    _, yaw_err, dim_err = best
    # Erro de altura: diferença entre as alturas das BASES (z = centro - height/2)
    z_err = ((est["center"][2] - est["dims"][2] / 2) - (gt["center"][2] - gt["dims"][2] / 2)) * 1000
    return {"pos_mm": pos_err, "yaw_deg": yaw_err, "dl_mm": dim_err[0], "dw_mm": dim_err[1],
            "dh_mm": dim_err[2], "dz_mm": z_err, "iou": iou_3d(est, gt)}


def evaluate_boxes(est_boxes, ground_truth):
    """Associa cada caixa estimada a uma verdadeira (máximo IoU, algoritmo húngaro) e calcula os erros.

    Para comparar no referencial do mundo (onde z é exatamente perpendicular à mesa), as caixas
    estimadas são levadas para o mundo com a pose VERDADEIRA da câmara 0 (só na avaliação).
    Caixas verdadeiras sem correspondência contam com IoU = 0.
    """
    T_world_camera0 = np.array(ground_truth["cameras"][0]["T_world_camera"])
    gt = [_box_in_world(np.array(b["T_world_box"]), b["length"], b["width"], b["height"])
          for b in ground_truth["boxes"]]
    est = [_box_in_world(T_world_camera0 @ b["T_camera0_box"], b["length"], b["width"], b["height"])
           for b in est_boxes]

    iou = np.array([[iou_3d(e, g) for g in gt] for e in est]).reshape(len(est), len(gt))
    rows_est, cols_gt = linear_sum_assignment(-iou) if len(est) else ([], [])
    results = []
    matched = {}
    for i, j in zip(rows_est, cols_gt):
        matched[j] = box_errors(est[i], gt[j])
    for j in range(len(gt)):
        results.append(matched.get(j, {"pos_mm": np.nan, "yaw_deg": np.nan, "dl_mm": np.nan,
                                       "dw_mm": np.nan, "dh_mm": np.nan, "dz_mm": np.nan,
                                       "iou": 0.0}))
    return results


# --------------------------------------------------------------------------------------
# Segmentação
# --------------------------------------------------------------------------------------
def inside_gt_boxes(points_camera0, ground_truth, margin=0.005):
    """Para cada caixa verdadeira, máscara dos pontos (referencial da câmara 0) que estão dentro dela
    (com uma margem de 5 mm por causa do ruído)."""
    masks = []
    for b in ground_truth["boxes"]:
        T_box_camera0 = np.linalg.inv(np.array(b["T_camera0_box"]))
        local = transform_points(T_box_camera0, points_camera0)
        half = np.array([b["length"], b["width"], b["height"]]) / 2 + margin
        masks.append(np.all(np.abs(local) <= half, axis=1))
    return masks


def evaluate_segmentation(points_table, labels, T_camera0_table, all_points_camera0, ground_truth, voxel=0.004):
    """Precisão e recall dos conjuntos de pontos de cada caixa.

    points_table, labels: pontos segmentados (referencial da mesa) e caixa atribuída a cada um.
    all_points_camera0: TODOS os pontos da nuvem agregada (para saber quantos pontos tem cada caixa
                        verdadeira, incluindo os que a segmentação deitou fora).
    As duas nuvens têm densidades diferentes (uma foi reduzida), por isso o recall é calculado
    com voxels: fração dos voxels ocupados pela caixa verdadeira que ficaram no conjunto certo.
    """
    pts_c0 = transform_points(T_camera0_table, points_table)
    gt_masks_seg = inside_gt_boxes(pts_c0, ground_truth)
    gt_masks_all = inside_gt_boxes(all_points_camera0, ground_truth)

    def voxel_keys(p):
        return set(map(tuple, np.floor(p / voxel).astype(np.int64)))

    num_sets = len(set(labels[labels >= 0]))
    # Associação conjunto <-> caixa verdadeira pelo maior número de pontos em comum
    overlap = np.array([[np.sum((labels == s) & m) for m in gt_masks_seg] for s in range(num_sets)])
    rows, cols = linear_sum_assignment(-overlap) if num_sets else ([], [])
    results = []
    for s, j in zip(rows, cols):
        mine = labels == s
        precision = np.sum(mine & gt_masks_seg[j]) / max(np.sum(mine), 1)
        gt_keys = voxel_keys(all_points_camera0[gt_masks_all[j]])
        my_keys = voxel_keys(pts_c0[mine & gt_masks_seg[j]])
        recall = len(gt_keys & my_keys) / max(len(gt_keys), 1)
        results.append({"gt_box": int(j), "precision": precision, "recall": recall})
    return num_sets, len(ground_truth["boxes"]), results


# #####################################################################################################
# 7. VISUALIZAÇÃO (janelas 3D do Open3D e figuras PNG do matplotlib)
# #####################################################################################################
# Janelas só com a opção --visualize; figuras com --figures, --evaluate_scenes e no run_test.py.

CAMERA_COLORS = plt.cm.tab10(np.arange(10))[:, :3]
BOX_COLORS = [(0.9, 0.1, 0.1), (0.1, 0.4, 0.9), (0.1, 0.7, 0.2), (0.9, 0.6, 0.0), (0.6, 0.2, 0.8), (0.0, 0.7, 0.7)]


# --------------------------------------------------------------------------------------
# Open3D
# --------------------------------------------------------------------------------------
def box_lineset(T, length, width, height, color):
    """LineSet com as 12 arestas de uma caixa com pose T."""
    corners, edges = box_lines_3d(T, length, width, height)
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(corners), o3d.utility.Vector2iVector(edges))
    ls.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return ls


def params_linesets(params, color=None):
    """LineSets das caixas (parâmetros no referencial da mesa)."""
    return [box_lineset(box_T_table(p[0], p[1], p[2], p[5], p[6]), p[3], p[4], p[5],
                        color or BOX_COLORS[j % len(BOX_COLORS)]) for j, p in enumerate(params)]


def show_segmentation(seg, labels):
    """Pontos dos objetos coloridos pela caixa a que foram atribuídos."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seg.points))
    pc.colors = o3d.utility.Vector3dVector([BOX_COLORS[l % len(BOX_COLORS)] if l >= 0 else (0.5, 0.5, 0.5)
                                            for l in labels])
    o3d.visualization.draw_geometries([pc], window_name="Segmentação (cor = caixa)")


def animate_optimization(points, colors, problem, delay=0.15):
    """Mostra a evolução das caixas ao longo da otimização, sobre a nuvem de pontos."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window("Evolução da otimização")
    vis.add_geometry(pc)
    current = []
    for x in evolution_snapshots(problem.history):
        for g in current:
            vis.remove_geometry(g, reset_bounding_box=False)
        current = params_linesets(x.reshape(-1, NUM_PARAMS))
        for g in current:
            vis.add_geometry(g, reset_bounding_box=False)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(delay)
    vis.run()                    # fica aberta no estado final até o utilizador fechar
    vis.destroy_window()


def show_final(points, colors, params_opt, params_bbox=None, gt_params=None):
    """Caixas finais: otimização (cores por caixa), bounding box (cinzento), verdadeiras (preto)."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.colors = o3d.utility.Vector3dVector(colors)
    geoms = [pc] + params_linesets(params_opt)
    if params_bbox is not None:
        geoms += params_linesets(params_bbox, (0.5, 0.5, 0.5))
    if gt_params is not None:
        geoms += params_linesets(gt_params, (0.0, 0.0, 0.0))
    o3d.visualization.draw_geometries(geoms, window_name="Caixas finais")


# --------------------------------------------------------------------------------------
# Figuras (matplotlib, vista de cima no referencial da mesa)
# --------------------------------------------------------------------------------------
def _draw_rects(ax, params, style, label=None):
    for j, p in enumerate(params):
        c = box_corners_2d(p[0], p[1], p[2], p[3], p[4])
        c = np.vstack([c, c[:1]])
        color = style.get("color") or BOX_COLORS[j % len(BOX_COLORS)]
        ax.plot(c[:, 0], c[:, 1], color=color, lw=style.get("lw", 1.5), ls=style.get("ls", "-"),
                label=label if j == 0 else None)


def save_top_view(path, points, colors, params_opt, params_bbox=None, gt_params=None, title=""):
    """Vista de cima: pontos + caixas (otimização, bounding box e, se houver, verdadeiras)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sel = np.random.default_rng(0).choice(len(points), min(len(points), 15000), replace=False)
    ax.scatter(points[sel, 0], points[sel, 1], c=colors[sel], s=1)
    if gt_params is not None:
        _draw_rects(ax, gt_params, {"color": "k", "lw": 2.5}, "verdadeira")
    if params_bbox is not None:
        _draw_rects(ax, params_bbox, {"color": "gray", "ls": "--"}, "bounding box")
    _draw_rects(ax, params_opt, {}, "otimização")
    ax.set_aspect("equal")
    ax.set_xlabel("x mesa (m)")
    ax.set_ylabel("y mesa (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_evolution(path, points, problem, num_frames=6):
    """Várias fases da otimização lado a lado (da estimativa inicial ao resultado final)."""
    snaps = evolution_snapshots(problem.history)
    picks = np.unique(np.linspace(0, len(snaps) - 1, min(num_frames, len(snaps))).astype(int))
    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.4), squeeze=False)
    sel = np.random.default_rng(0).choice(len(points), min(len(points), 6000), replace=False)
    for ax, k in zip(axes[0], picks):
        ax.scatter(points[sel, 0], points[sel, 1], c="0.6", s=0.5)
        params = [normalize_box(p) for p in snaps[k].reshape(-1, NUM_PARAMS)]
        _draw_rects(ax, params, {})
        ax.set_aspect("equal")
        ax.set_title(f"passo {k}/{len(snaps) - 1}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Evolução das caixas durante a otimização (vista de cima)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# #####################################################################################################
# 8. AVALIAÇÃO EM VÁRIAS CENAS (opção --evaluate_scenes: gera as tabelas do Read_me)
# #####################################################################################################
ROOT_FOLDER = Path(__file__).resolve().parent.parent      # pasta principal do trabalho


def fmt(v, d=1):
    """Número com d casas decimais, ou "-" se não existir (para as tabelas)."""
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def evaluate_scenes(patterns, gt_registration=False, use_color=True, output=None):
    """Corre VÁRIAS cenas com ground truth e escreve as tabelas de erros dos dois métodos (para o Read_me).

    patterns: pastas das cenas (aceita wildcards, por exemplo scenes_offline/level_*). O número de
    caixas é lido do config.json da cena, o que é permitido porque isto só serve para avaliar.
    gt_registration=True usa as poses verdadeiras das câmaras (avalia só a segmentação e o ajuste).
    Escreve <output>/results.md (tabelas em Markdown) e duas figuras por cena. As cenas são
    processadas uma de cada vez.
    """
    folders = sorted({f for pattern in patterns for f in glob.glob(pattern) if Path(f, "ground_truth.json").exists()})
    out = Path(output) if output else ROOT_FOLDER / "results_offline"
    out.mkdir(parents=True, exist_ok=True)
    if not folders:
        print("Nenhuma cena com ground_truth.json encontrada.")
        return

    reg_lines = ["| cena | erro transl. médio (mm) | máx (mm) | erro rot. médio (°) | máx (°) | só encadeamento: transl. máx (mm) |",
                 "|---|---|---|---|---|---|"]
    seg_lines = ["| cena | conjuntos / caixas | precisão média | recall médio |", "|---|---|---|---|"]
    box_lines = ["| cena | caixa | método | pos (mm) | yaw (°) | dL (mm) | dW (mm) | dH (mm) | dz (mm) | IoU |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
    summary = ["| cena | IoU médio bounding box | IoU médio otimização |", "|---|---|---|"]

    for folder in folders:
        num_boxes = len(json.loads(Path(folder, "config.json").read_text(encoding="utf-8"))["boxes"])
        name = Path(folder).name
        detection = Path(folder) / "detecao"
        det.run_detection(folder, detection, gt_registration=gt_registration)       # script 1
        r = run(detection, num_boxes, use_color=use_color)                          # script 2
        gt = r["ground_truth"]

        if not gt_registration:
            cams = camera_errors(r["T_cams"], gt)
            chain = camera_errors(r["T_chain"], gt)
            t, a = np.array([c[1] for c in cams]), np.array([c[2] for c in cams])
            reg_lines.append(f"| {name} | {t[1:].mean():.2f} | {t.max():.2f} | {a[1:].mean():.3f} | {a.max():.3f} | "
                             f"{max(c[1] for c in chain):.2f} |")

        n_sets, n_gt, seg_rows = evaluate_segmentation(
            r["seg"].points, r["box_labels"], r["table"].T_camera0_table, full_cloud_points(r), gt)
        seg_lines.append(f"| {name} | {n_sets} / {n_gt} | {np.mean([s['precision'] for s in seg_rows]):.3f} | "
                         f"{np.mean([s['recall'] for s in seg_rows]):.3f} |")

        ious = {}
        for method, label in (("bbox", "bounding box"), ("opt", "otimização")):
            rows = evaluate_boxes(boxes_to_camera0(r[f"params_{method}"], r["table"].T_camera0_table), gt)
            ious[method] = np.mean([e["iou"] for e in rows])
            for j, e in enumerate(rows):
                box_lines.append(f"| {name} | {j} | {label} | {fmt(e['pos_mm'])} | {fmt(e['yaw_deg'], 2)} | "
                                 f"{fmt(e['dl_mm'])} | {fmt(e['dw_mm'])} | {fmt(e['dh_mm'])} | {fmt(e['dz_mm'])} | {e['iou']:.3f} |")
        summary.append(f"| {name} | {ious['bbox']:.3f} | {ious['opt']:.3f} |")

        gt_params = gt_params_in_table(gt, r["table"].T_camera0_table)
        save_top_view(out / f"{name}_top_view.png", r["seg"].points, r["seg"].colors,
                                    r["params_opt"], r["params_bbox"], gt_params, title=name)
        save_evolution(out / f"{name}_optimization.png", r["fit_points"], r["problem"])

    text = ["# Resultados", "",
            f"Registo: {'poses verdadeiras (só segmentação e ajuste)' if gt_registration else 'estimado'}", ""]
    if not gt_registration:
        text += ["## Registo", ""] + reg_lines + [""]
    text += ["## Segmentação", ""] + seg_lines + [""]
    text += ["## Resumo (IoU 3D médio)", ""] + summary + [""]
    text += ["## Erros por caixa", ""] + box_lines + [""]
    (out / "results.md").write_text("\n".join(text), encoding="utf-8")
    print(f"\nTabelas escritas em {out / 'results.md'}")


# #####################################################################################################
# 9. PROGRAMA PRINCIPAL
# #####################################################################################################
def load_detection(folder):
    """Lê a pasta escrita pelo script 1: a nuvem só das caixas (caixas.ply) e o detecao.json.

    Saída: dicionário com os pontos e as cores das caixas (referencial da mesa), a pose de cada câmara
    (grafo de poses e só encadeamento), a mesa (Table), os nomes das imagens, a pasta da cena e o
    ground truth (None se a cena não o tiver; só serve para avaliar, NUNCA para estimar).
    """
    folder = Path(folder)
    info = json.loads((folder / det.DETECTION_FILE).read_text(encoding="utf-8"))
    cloud = o3d.io.read_point_cloud(str(folder / info["nuvem_das_caixas"]))
    mesa = info["mesa"]
    table = Table(np.array(mesa["T_camera0_table"]), mesa["comprimento"], mesa["largura"], np.array(mesa["normal"]))
    gt_path = Path(info["cena"]) / "ground_truth.json"
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else None
    return {"points": np.asarray(cloud.points), "colors": np.asarray(cloud.colors),
            "T_cams": [np.array(T) for T in info["T_camera0_camera"]],
            "T_chain": [np.array(T) for T in info["T_camera0_camera_so_encadeamento"]],
            "table": table, "images": info["imagens"], "scene_folder": Path(info["cena"]),
            "ground_truth": ground_truth}


def full_cloud_points(r):
    """TODOS os pontos da nuvem agregada (referencial da câmara 0), só para o recall da segmentação.

    O script 1 só grava os pontos das caixas; para saber quantos pontos tem cada caixa verdadeira
    (incluindo os que a segmentação deitou fora) volta-se a juntar as nuvens da cena com as poses
    das câmaras já estimadas (não se repete o registo).
    """
    cloud, _ = aggregate(load_scene(r["scene_folder"]), r["T_cams"])
    return np.asarray(cloud.points)


def ask_num_boxes():
    """Pergunta no terminal o número de caixas da cena."""
    while True:
        answer = input("Número de caixas na mesa: ").strip()
        if answer.isdigit() and int(answer) > 0:
            return int(answer)
        print("  escreve um número inteiro positivo")


def run(detection_folder, num_boxes, use_color=True, verbose=True):
    """Parte 2 (separar as caixas) e Parte 3 (ajuste) a partir da deteção do script 1.

    Entrada: pasta da deteção (caixas.ply + detecao.json) e o número de caixas.
    Saída: dicionário com os resultados de cada etapa (para o result.json, a avaliação e as figuras).
    """
    t0 = time.time()
    r = load_detection(detection_folder)
    if verbose:
        print(f"Deteção {detection_folder}: {len(r['points'])} pontos das caixas, {num_boxes} caixas")

    # ---------------- Parte 2: separar as caixas ----------------
    seg = segment(r["points"], r["colors"], num_boxes, use_color=use_color, verbose=verbose)

    # ---------------- Parte 3: ajuste ----------------
    # Multi-start: com caixas da mesma cor em contacto a estimativa inicial é incerta e a otimização
    # pode ficar presa num mínimo local. Experimentam-se duas estimativas iniciais (k-means e degraus
    # do contorno) e fica a que acaba com MENOR CUSTO (os custos são comparáveis: a função objetivo e
    # os pontos são os mesmos nas duas tentativas).
    methods = ["kmeans", "profile"] if needs_geometric_split(seg) else ["kmeans"]
    # Centros das câmaras no referencial da mesa (para o ajuste saber que faces podem ser vistas)
    T_table_camera0 = np.linalg.inv(r["table"].T_camera0_table)
    camera_centers = np.array([(T_table_camera0 @ T)[:3, 3] for T in r["T_cams"]])
    best = None
    for method in methods:
        init_params, init_colors, box_labels = initial_boxes(seg, method)
        params_opt, problem, fit_points, fit_assoc = fit_boxes(seg, init_params, init_colors,
                                                               use_color=use_color,
                                                               camera_centers=camera_centers, verbose=verbose)
        cost = min(h[1] for h in problem.history)
        if verbose and len(methods) > 1:
            print(f"    estimativa inicial '{method}': custo final {cost:.1f}")
        if best is None or cost < best[0]:
            best = (cost, init_params, init_colors, box_labels, params_opt, problem, fit_points, fit_assoc)
    _, init_params, init_colors, box_labels, params_opt, problem, fit_points, fit_assoc = best

    # Método base, para comparação
    params_bbox = bbox_method(seg.points, box_labels, len(init_params), init_params=init_params)
    if verbose:
        print(f"  tempo: {time.time() - t0:.1f} s")

    r.update({"seg": seg, "init_params": np.array(init_params), "box_labels": box_labels,
              "params_bbox": params_bbox, "params_opt": params_opt, "problem": problem,
              "fit_points": fit_points, "fit_assoc": fit_assoc})
    return r


def main():
    """Lê os argumentos da linha de comandos, ajusta as caixas e escreve o result.json."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detection", help="pasta escrita pelo script 1 (ex.: test/cena_1/detecao)")
    parser.add_argument("--num_boxes", type=int, help="número de caixas na mesa (se faltar, é perguntado)")
    parser.add_argument("--output", help="ficheiro de saída (por omissão: <cena>/result.json)")
    parser.add_argument("--method", choices=["opt", "bbox"], default="opt",
                        help="método usado no result.json: otimização (opt) ou bounding box (bbox)")
    parser.add_argument("--no_color", action="store_true", help="não usar a cor (segmentação nem otimização)")
    parser.add_argument("--evaluate", action="store_true", help="compara com o ground_truth.json, se existir")
    parser.add_argument("--visualize", action="store_true", help="abre janelas 3D com cada etapa")
    parser.add_argument("--figures", action="store_true", help="grava top_view.png e optimization.png")
    parser.add_argument("--evaluate_scenes", nargs="+", metavar="CENA",
                        help="corre os 2 scripts em várias cenas com ground truth (aceita wildcards) e escreve as tabelas")
    parser.add_argument("--gt_registration", action="store_true",
                        help="com --evaluate_scenes: usa as poses verdadeiras (avalia só segmentação e ajuste)")
    args = parser.parse_args()

    # Modo "várias cenas": só avaliação, não escreve result.json
    if args.evaluate_scenes:
        evaluate_scenes(args.evaluate_scenes, gt_registration=args.gt_registration,
                        use_color=not args.no_color, output=args.output)
        return
    if args.detection is None:
        parser.error("indica --detection (a pasta escrita pelo script 1) ou --evaluate_scenes")
    num_boxes = args.num_boxes if args.num_boxes else ask_num_boxes()

    r = run(args.detection, num_boxes, use_color=not args.no_color)

    # Escrita do result.json
    params = r["params_opt"] if args.method == "opt" else r["params_bbox"]
    boxes = boxes_to_camera0(params, r["table"].T_camera0_table)
    output = Path(args.output) if args.output else r["scene_folder"] / "result.json"
    write_result(output, r["images"], r["T_cams"], boxes)
    print(f"\nResultado escrito em {output}")

    # Opções extra (secções 6 e 7)
    gt = r["ground_truth"]
    if args.evaluate or args.visualize or args.figures:
        gt_params = gt_params_in_table(gt, r["table"].T_camera0_table) if gt else None
        if args.figures:
            save_top_view(output.parent / "top_view.png", r["seg"].points, r["seg"].colors,
                          r["params_opt"], r["params_bbox"], gt_params, title=r["scene_folder"].name)
            save_evolution(output.parent / "optimization.png", r["fit_points"], r["problem"])
        if args.evaluate and gt:
            print_evaluation(r)
        if args.visualize:
            show_segmentation(r["seg"], r["box_labels"])
            animate_optimization(r["fit_points"], np.full_like(r["fit_points"], 0.6), r["problem"])
            show_final(r["seg"].points, r["seg"].colors, r["params_opt"], r["params_bbox"], gt_params)


if __name__ == "__main__":
    main()
