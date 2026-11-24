#!/usr/bin/env python3
"""SAVIBoxes - Trabalho 1 de SAVI. PARTE 3: AJUSTE DAS CAIXAS (otimização e bounding box -> result.json).

=====================================================================================================
COMO CORRER
=====================================================================================================
    (a partir da pasta principal do trabalho, DEPOIS das Partes 1 e 2)
    python savi/parte3_ajuste.py --scene test/cena_1
    python savi/parte3_ajuste.py --scene test/cena_1 --visualize   # janelas 3D: bounding box, otimização
                                                                   # e evolução da otimização (animação)

Lê <cena>/partes/ (registo.json da Parte 1; objetos.ply, mesa.json e segmentacao.npz da Parte 2),
ajusta as caixas pelos dois métodos do enunciado e escreve o result.json (por omissão, na pasta da cena)
com a pose de cada câmara e a pose e as dimensões de cada caixa, no referencial da câmara 000.
Grava também <cena>/partes/ajuste.npz (para o avaliador.py).

Ordem das partes (o run_test.py corre-as todas, cena a cena):
    parte1_registo.py      -> <cena>/partes/registo.json   (pose de cada câmara)
    extra_fotometria.py    -> <cena>/partes/fotometria.json (ganho e desvio de cada imagem)
    parte2_segmentacao.py  -> <cena>/partes/objetos.ply, mesa.json, segmentacao.npz (estima o n.º de caixas)
    parte3_ajuste.py       -> result.json e <cena>/partes/ajuste.npz
    avaliador.py           -> pede o n.º real de caixas e compara com o ground truth (se a cena o tiver)

=====================================================================================================
ORGANIZAÇÃO DESTE FICHEIRO
=====================================================================================================
    1. Otimização         estimativa inicial + modelo da caixa + função objetivo + least_squares
    2. Método base        bounding box orientada (só para comparar com a otimização)
    3. Saída              gravar e ler o ajuste.npz e escrever o result.json
    4. Visualização       janelas 3D: cena com as caixas da bounding box, com as da otimização e evolução
    5. Programa principal run_fit() e main()
(Convenções: caixa = [x, y, yaw, length, width, height, z] no referencial da mesa; ver comum.py.)
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2                                   # OpenCV: k-means, retângulos
import matplotlib.pyplot as plt              # figuras PNG vistas de cima (secção 4)
import numpy as np
import open3d as o3d                         # nuvens de pontos: bounding boxes, janelas 3D
from scipy.optimize import least_squares     # otimização (Parte 05/06)
from scipy.spatial import cKDTree            # procura rápida do vizinho mais próximo

from comum import (  # geometria e leitura da cena (comuns a todas as partes)
    box_corners_2d, box_lines_3d, box_signed_distance, box_T_table, boxes_to_camera0, load_ground_truth,
    load_scene, OUTPUT_FOLDER, transform_points, wrap_angle, write_result, yaw_of,
)
from parte1_registo import (  # funções da Parte 1 usadas aqui
    aggregate, load_registration,
)
from visualizacao import (  # janelas 3D (comuns a todas as partes)
    box_item, camera_view, close_view, colors_uint8, data_view, points_item, show_windows,
)
from parte2_segmentacao import (  # funções da Parte 2 usadas aqui
    chroma_features, count_top_levels, load_objects, load_segmentation, top_surface_points,
)


# #####################################################################################################
# 1. PARTE 3 - AJUSTE DAS CAIXAS POR OTIMIZAÇÃO
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
# 1.1 Estimativa inicial (Parte 06, Ex. 6: a otimização precisa de começar perto da solução)
# ----------------------------------------------------------------------------------------------------
# Funções de apoio para dividir um cluster em caixas (estavam na Parte 2, mas só a Parte 3 as usa)
def largest_component(points, eps=0.012):
    """Índices da maior componente ligada (DBSCAN) de um conjunto de pontos (retira pontos soltos)."""
    if len(points) < 20:
        return np.arange(len(points))
    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).cluster_dbscan(eps, 5))
    if labels.max() < 0:
        return np.arange(len(points))
    return np.nonzero(labels == np.bincount(labels[labels >= 0]).argmax())[0]


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


def allocate_boxes(points, groups, num_boxes):
    """Reparte num_boxes caixas pelos grupos de pontos (usado pela Parte 3 dentro de um cluster:
    quantas das caixas estimadas desse cluster ficam em cada grupo de cor).

    - Tantos grupos como caixas: uma caixa por grupo;
    - Menos grupos do que caixas: há caixas da mesma cor. As caixas extra vão primeiro para os grupos
      com mais TOPOS do que caixas (uma pilha tem pouca área vista de cima, mas tem vários topos). As
      que sobram vão, uma a uma, para o grupo com maior área por caixa;
    - Mais grupos do que caixas: ficam os maiores.
    Saída: dicionário grupo -> número de caixas.
    """
    labels = list(range(groups.max() + 1))[:num_boxes]
    counts = {k: 1 for k in labels}
    areas = {k: footprint_area(points[groups == k, :2]) for k in labels}
    tops = {k: count_top_levels(points[groups == k]) for k in labels} if num_boxes > len(labels) else {}
    for _ in range(num_boxes - len(labels)):
        missing = [c for c in labels if tops[c] > counts[c]]
        if missing:
            k = max(missing, key=lambda c: areas[c])
        else:
            k = max(labels, key=lambda c: areas[c] / counts[c])
        counts[k] += 1
    return counts


def rect_params(points_xy):
    """Retângulo de área mínima de pontos 2D (cv2.minAreaRect) -> [x, y, yaw, length, width]."""
    (cx, cy), (a, b), angle = cv2.minAreaRect(points_xy.astype(np.float32))
    return np.array([cx, cy, np.radians(angle), max(a, 0.02), max(b, 0.02)])


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
# 1.2 Modelo da caixa
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
# 1.3 Função objetivo e otimização
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
# 2. MÉTODO BASE: BOUNDING BOX ORIENTADA (só para comparar com a otimização)
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
# 3. SAÍDA: ajuste.npz (para o avaliador) e result.json
# #####################################################################################################
FIT_FILE = "ajuste.npz"     # caixas dos dois métodos, estimativa inicial, caixa de cada ponto, evolução


def save_fit(folder, r3):
    """Grava o ajuste (ajuste.npz): caixas da otimização e da bounding box, estimativa inicial, caixa
    atribuída a cada ponto da segmentação e as caixas em cada passo da otimização."""
    np.savez_compressed(Path(folder) / FIT_FILE, params_opt=r3["params_opt"],
                        params_bbox=np.asarray(r3["params_bbox"]).reshape(-1, NUM_PARAMS),
                        init_params=r3["init_params"], box_labels=r3["box_labels"],
                        snapshots=np.array(evolution_snapshots(r3["problem"].history)))


def load_fit(folder):
    """Lê o ajuste.npz. Saída: dicionário com as mesmas chaves que run_fit (sem o problema de otimização)."""
    d = np.load(Path(folder) / FIT_FILE)
    return {k: d[k] for k in d.files}


def write_scene_result(path, r1, r2, r3, method="opt"):
    """Escreve o result.json: pose de cada câmara (Parte 1) e as caixas do método escolhido (otimização
    ou bounding box), passadas do referencial da mesa para o da câmara 000."""
    params = r3["params_opt"] if method == "opt" else r3["params_bbox"]
    boxes = boxes_to_camera0(params, r2["table"].T_camera0_table)
    return write_result(path, r1["images"], r1["T_cams"], boxes)


# #####################################################################################################
# 4. VISUALIZAÇÃO
# #####################################################################################################
BOX_COLORS = [(0.9, 0.1, 0.1), (0.1, 0.4, 0.9), (0.1, 0.7, 0.2), (0.9, 0.6, 0.0), (0.6, 0.2, 0.8), (0.0, 0.7, 0.7)]


def gt_params_in_table(ground_truth, T_camera0_table):
    """Caixas verdadeiras no referencial da mesa estimada (só para desenhar nas figuras)."""
    T_table_camera0 = np.linalg.inv(T_camera0_table)
    params = []
    for b in ground_truth["boxes"]:
        T = T_table_camera0 @ np.array(b["T_camera0_box"])
        z_base = T[2, 3] - b["height"] / 2               # altura da base acima da mesa
        params.append([T[0, 3], T[1, 3], yaw_of(T[:3, :3]), b["length"], b["width"], b["height"], z_base])
    return np.array(params)


# ----------------------------------------------------------------------------------------------------
# Desenho das caixas vistas de cima (figuras PNG)
# ----------------------------------------------------------------------------------------------------
def _draw_rects(ax, params, style, label=None):
    for j, p in enumerate(params):
        c = box_corners_2d(p[0], p[1], p[2], p[3], p[4])
        c = np.vstack([c, c[:1]])
        color = style.get("color") or BOX_COLORS[j % len(BOX_COLORS)]
        ax.plot(c[:, 0], c[:, 1], color=color, lw=style.get("lw", 1.5), ls=style.get("ls", "-"),
                label=label if j == 0 else None)


# ----------------------------------------------------------------------------------------------------
# Figuras PNG (vista de cima), usadas pelo avaliador.py --evaluate_scenes
# ----------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------
# Janelas 3D do Open3D: a cena com as caixas da bounding box, com as caixas da otimização e a evolução
# da otimização (animação)
# ----------------------------------------------------------------------------------------------------
MAX_FRAMES = 60             # passos da otimização mostrados na animação
GT_COLOR = (0.0, 0.0, 0.0)  # caixas verdadeiras (se houver ground truth): a preto


def snapshot_frames(snapshots, max_frames=MAX_FRAMES):
    """Até max_frames passos da otimização, igualmente espaçados (com o primeiro e o último), cada um como
    uma lista de caixas [x, y, yaw, l, w, h, z]. Saída: (caixas de cada passo, número de cada passo)."""
    snapshots = np.asarray(snapshots)
    picks = np.unique(np.linspace(0, len(snapshots) - 1, min(max_frames, len(snapshots))).astype(int))
    frames = np.array([[normalize_box(p) for p in snapshots[k].reshape(-1, NUM_PARAMS)] for k in picks])
    return frames, picks


def figure_data(r1, r2, r3, gt_params=None):
    """Dados das janelas da Parte 3 (r1, r2 = Partes 1 e 2, r3 = run_fit ou load_fit), no referencial da
    mesa: a nuvem agregada (se existir; senão só os pontos dos objetos), os pontos dos objetos, as caixas
    dos dois métodos, as caixas ao longo da otimização e, se houver ground truth, as verdadeiras."""
    table, seg = r2["table"], r2["seg"]
    T_table_camera0 = np.linalg.inv(table.T_camera0_table)
    if r2.get("cloud") is not None:
        points = transform_points(T_table_camera0, np.asarray(r2["cloud"].points))
        colors = r2["cloud"].colors
    else:
        points, colors = seg.points, seg.colors
    snapshots = evolution_snapshots(r3["problem"].history) if "problem" in r3 else r3["snapshots"]
    frames, steps = snapshot_frames(snapshots)
    center = seg.points.mean(axis=0)
    view = camera_view(T_table_camera0 @ r1["T_cams"][0], center, np.array([0.0, 0.0, 1.0]), back=-0.3, above=0.1)
    return {"parte": "parte3_ajuste", "cena": Path(r2["scene_folder"]).name,
            "points": points.astype(np.float32), "colors": colors_uint8(colors),
            "objects": seg.points.astype(np.float32), "object_colors": colors_uint8(seg.colors),
            "params_bbox": np.asarray(r3["params_bbox"]).reshape(-1, NUM_PARAMS),
            "params_opt": np.asarray(r3["params_opt"]).reshape(-1, NUM_PARAMS), "frames": frames,
            "steps": steps, "total_steps": len(snapshots) - 1,
            "gt_params": np.zeros((0, NUM_PARAMS)) if gt_params is None else np.asarray(gt_params),
            **{f"view_{k}": v for k, v in view.items()}}


def top_view(points):
    """Ponto de vista de cima (referencial da mesa): por cima do centro dos pontos dos objetos, a olhar para
    baixo, com x para a direita e y para cima (como uma figura vista de cima)."""
    center = (points.min(axis=0) + points.max(axis=0)) / 2       # centro da caixa que envolve os pontos
    height = max(0.25, 1.3 * float(np.ptp(points[:, :2], axis=0).max()))
    return {"center": center, "eye": center + np.array([0.0, 0.0, height]), "up": np.array([0.0, 1.0, 0.0])}


def make_windows(data):
    """Vistas da Parte 3 (caixas verdadeiras a preto, se houver ground truth):
    1. método base (bounding box) visto de cima, sobre os pontos dos objetos;
    2. método principal (otimização) visto de cima;
    3. a cena em 3D com as caixas da bounding box;
    4. a cena em 3D com as caixas da otimização;
    5. a evolução das caixas ao longo da otimização em 3D, sobre os pontos dos objetos (animação; a
       figura gravada mostra o último passo)."""
    gt = [box_item(p, GT_COLOR, 2.0) for p in data["gt_params"]]
    note = " (verdadeiras a preto)" if len(data["gt_params"]) else ""
    title = f"Parte 3 - {data['cena']}"
    view = data_view(data)
    objects = points_item(data["objects"], data["object_colors"], 3.0)
    methods = (("bounding_box", "params_bbox", "método base: bounding box"),
               ("otimizacao", "params_opt", "método principal: otimização"))
    boxes = {key: [box_item(p, BOX_COLORS[j % len(BOX_COLORS)]) for j, p in enumerate(data[key])]
             for _, key, _ in methods}
    windows = []
    for name, key, method in methods:                     # vistas de cima
        windows.append({"name": f"_{name}_vista_de_cima",
                        "title": f"{title}: {method}, vista de cima ({len(data[key])} caixas){note}",
                        "items": [objects] + gt + boxes[key], "view": top_view(data["objects"])})
    for name, key, method in methods:                     # na cena, em 3D
        windows.append({"name": f"_{name}_3d", "title": f"{title}: {method}, em 3D na cena ({len(data[key])} caixas){note}",
                        "items": [points_item(data["points"], data["colors"])] + gt + boxes[key], "view": view})
    frames = [[box_item(p, BOX_COLORS[j % len(BOX_COLORS)]) for j, p in enumerate(frame)] for frame in data["frames"]]
    windows.append({"name": "_evolucao",
                    "title": f"{title}: evolução da otimização em 3D ({int(data['total_steps'])} passos, animação){note}",
                    "items": [objects] + gt, "view": close_view(view, data["objects"]), "frames": frames})
    return windows


# ----------------------------------------------------------------------------------------------------
# Janelas antigas do Open3D (opção --open3d)
# ----------------------------------------------------------------------------------------------------
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


# #####################################################################################################
# 5. PROGRAMA PRINCIPAL
# #####################################################################################################
def run_fit(r1, r2, use_color=True, verbose=True):
    """Parte 3 numa cena: estimativa inicial, ajuste por otimização e bounding box.

    Entrada: o resultado da Parte 1 (poses das câmaras) e da Parte 2 (mesa e segmentação).
    Saída: dicionário com as caixas dos dois métodos (referencial da mesa), a estimativa inicial, a caixa
    de cada ponto, o problema de otimização (com o histórico) e os pontos usados. Grava o ajuste.npz.
    """
    t0 = time.time()
    seg, table = r2["seg"], r2["table"]

    # Multi-start: com caixas da mesma cor em contacto a estimativa inicial é incerta e a otimização
    # pode ficar presa num mínimo local. Experimentam-se duas estimativas iniciais (k-means e degraus
    # do contorno) e fica a que acaba com MENOR CUSTO (os custos são comparáveis: a função objetivo e
    # os pontos são os mesmos nas duas tentativas).
    methods = ["kmeans", "profile"] if needs_geometric_split(seg) else ["kmeans"]
    # Centros das câmaras no referencial da mesa (para o ajuste saber que faces podem ser vistas)
    T_table_camera0 = np.linalg.inv(table.T_camera0_table)
    # (só as câmaras que entraram na nuvem: uma câmara com o registo falhado está no sítio errado)
    camera_centers = np.array([(T_table_camera0 @ T)[:3, 3] for T, ok in zip(r1["T_cams"], r1["registered"]) if ok])
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

    r3 = {"init_params": np.array(init_params), "box_labels": box_labels, "params_bbox": params_bbox,
          "params_opt": params_opt, "problem": problem, "fit_points": fit_points, "fit_assoc": fit_assoc}
    save_fit(r1["folder"], r3)
    return r3


def main():
    """Lê os argumentos da linha de comandos, ajusta as caixas e escreve o result.json."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="pasta da cena (já com as Partes 1 e 2 feitas)")
    parser.add_argument("--output", help="ficheiro de saída (por omissão: <cena>/result.json)")
    parser.add_argument("--method", choices=["opt", "bbox"], default="opt",
                        help="método usado no result.json: otimização (opt) ou bounding box (bbox)")
    parser.add_argument("--no_color", action="store_true", help="não usar a cor na otimização")
    parser.add_argument("--visualize", action="store_true",
                        help="janelas 3D: bounding box, otimização e evolução da otimização (animação)")
    parser.add_argument("--open3d", action="store_true", help="janelas antigas do Open3D (segmentação, evolução, final)")
    args = parser.parse_args()

    folder = Path(args.scene) / OUTPUT_FOLDER
    r1 = load_registration(folder)
    table, points, colors = load_objects(folder)
    r2 = {"table": table, "object_points": points, "object_colors": colors, "seg": load_segmentation(folder),
          "scene_folder": r1["scene_folder"]}
    r3 = run_fit(r1, r2, use_color=not args.no_color)

    output = Path(args.output) if args.output else Path(r1["scene_folder"]) / "result.json"
    write_scene_result(output, r1, r2, r3, args.method)
    print(f"\nResultado escrito em {output}")

    # Caixas verdadeiras (se a cena tiver ground truth): só para desenhar
    gt = load_ground_truth(r1["scene_folder"])
    gt_params = gt_params_in_table(gt, table.T_camera0_table) if gt else None
    if args.visualize:
        # a nuvem agregada (só para desenhar a cena à volta das caixas)
        r2["cloud"], _ = aggregate(load_scene(r1["scene_folder"]), r1["T_cams"], r1["registered"])
        show_windows(make_windows(figure_data(r1, r2, r3, gt_params)))
    if args.open3d:
        seg = r2["seg"]
        show_segmentation(seg, r3["box_labels"])
        animate_optimization(r3["fit_points"], np.full_like(r3["fit_points"], 0.6), r3["problem"])
        show_final(seg.points, seg.colors, r3["params_opt"], r3["params_bbox"], gt_params)


if __name__ == "__main__":
    main()
