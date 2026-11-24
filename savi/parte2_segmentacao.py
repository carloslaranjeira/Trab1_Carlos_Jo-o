#!/usr/bin/env python3
"""SAVIBoxes - Trabalho 1 de SAVI. PARTE 2: SEGMENTAÇÃO (mesa, pontos dos objetos e separação das caixas).

=====================================================================================================
COMO CORRER
=====================================================================================================
    (a partir da pasta principal do trabalho, DEPOIS da Parte 1: python savi/parte1_registo.py --scene ...)
    python savi/parte2_segmentacao.py --scene test/cena_1
    python savi/parte2_segmentacao.py --scene test/cena_1 --visualize   # janelas 3D

A segmentação é feita sobre a nuvem AGREGADA (as nuvens são juntadas com as poses do registo.json).
Resultado esperado: os pontos dos objetos sobre a mesa ficam isolados do resto da nuvem agregada, que
fica a cinzento (primeira janela do --visualize).
Grava em <cena>/partes/: objetos.ply (nuvem de pontos SÓ DOS OBJETOS, com cor, no referencial da mesa),
mesa.json (pose e dimensões da mesa) e segmentacao.npz (cluster e grupo de cor de cada ponto).

Ordem das partes (o run_test.py corre-as todas, cena a cena):
    parte1_registo.py      -> <cena>/partes/registo.json   (pose de cada câmara)
    extra_fotometria.py    -> <cena>/partes/fotometria.json (ganho e desvio de cada imagem)
    parte2_segmentacao.py  -> <cena>/partes/objetos.ply, mesa.json, segmentacao.npz (estima o n.º de caixas)
    parte3_ajuste.py       -> result.json e <cena>/partes/ajuste.npz
    avaliador.py           -> pede o n.º real de caixas e compara com o ground truth (se a cena o tiver)

=====================================================================================================
ORGANIZAÇÃO DESTE FICHEIRO (as 4 subpartes do enunciado, pela mesma ordem)
=====================================================================================================
    1. Mesa e referencial   chão e mesa com RANSAC guiado pelas normais, referencial da mesa
                            (Parte 08, Ex. 4; Parte 07, Ex. 2, 5, 6 e 8)
    2. Retirar mesa e fundo pontos acima da mesa e dentro dos limites do tampo (Parte 08, Ex. 4)
    3. Clustering           clustering euclidiano (DBSCAN) dos objetos, e as figuras da Parte 2
                            (Parte 07, Ex. 4; Parte 08, Ex. 4)
    4. Cor                  separar caixas em contacto pela cor (Nível 4; ideia do HSV da Parte 02)
    5. Extra                número de caixas desconhecido: estimar quantas caixas tem cada cluster
                            (alturas, cores, desvio lateral e rotações; normais da Parte 07, Ex. 5-7)
    6. Programa principal   segment(), gravar e ler objetos.ply / mesa.json / segmentacao.npz,
                            run_segmentation() e main()
(As convenções - T_A_B, referencial da mesa - estão em comum.py.)
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2                                   # OpenCV: k-means, retângulo de área mínima
import numpy as np
import open3d as o3d                         # nuvens de pontos: RANSAC de planos, DBSCAN
from scipy.spatial import cKDTree            # procura rápida do vizinho mais próximo

from comum import (  # geometria e leitura da cena (comuns a todas as partes)
    box_corners_2d, load_scene, make_T, OUTPUT_FOLDER, transform_points,
)
from parte1_registo import (  # funções da Parte 1 usadas aqui
    aggregate, load_registration,
)
from visualizacao import (  # janelas 3D (comuns a todas as partes)
    CAMERA_COLORS, camera_view, close_view, colors_uint8, data_view, frame_item, lines_item, points_item,
    show_windows,
)


# Parâmetros (valores escolhidos com testes nas cenas do simulador)
PLANE_DISTANCE = 0.008       # 1. m: distância máxima ao plano para um ponto ser do plano (RANSAC)
HORIZONTAL_MAX_ANGLE = 10    # 1. graus: normal a menos disto da vertical -> ponto de uma superfície horizontal
TABLE_HEIGHT_MIN = 0.5       # 1. m: o tampo está entre 0.5 m e 1.0 m acima do chão (Parte 08, Ex. 4)
TABLE_HEIGHT_MAX = 1.0
MIN_PLANE_POINTS = 300       # 1. n.º mínimo de pontos (voxel de 1 cm) para aceitar um plano do chão ou da mesa
ABOVE_TABLE_MIN = 0.008      # 2. m: pontos abaixo disto (ruído do tampo) são retirados
ABOVE_TABLE_MAX = 0.60       # 2. m: pontos acima disto já não são objetos sobre a mesa (pilhas até 60 cm)
TABLE_MARGIN = 0.01          # 2. m: margem para dentro dos limites do tampo
OBJECT_VOXEL = 0.004         # 2. m: voxel usado nos pontos dos objetos
CLUSTER_EPS = 0.015          # 3. m: no DBSCAN, pontos a menos de 1.5 cm ficam no mesmo cluster
WALL_MAX_NZ = 0.3            # 5. |n_z| máximo de um ponto de parede de caixa (normal quase horizontal)
MIN_ROTATION = 10            # 5. graus: paredes com orientações mais próximas do que isto são da mesma caixa
MIN_WALL_FRACTION = 0.10     # 5. fração mínima das paredes de um cluster numa orientação para contar como caixa
LATERAL_STEP = 0.005         # 5. m: salto do centro do topo (desvio lateral) que marca uma junta entre caixas
                             #    (numa caixa isolada o salto medido nunca passou de 2.3 mm; com 4 mm já
                             #    apareciam juntas falsas)
WIDTH_STEP = 0.015           # 5. m: salto da largura do topo (larguras diferentes) que marca uma junta
MIN_BOX_SIDE = 0.05          # 5. m: lado mínimo de uma caixa (juntas mais próximas do que isto são a mesma)


@dataclass
class Table:
    """Mesa detetada."""
    T_camera0_table: np.ndarray   # pose do referencial da mesa no referencial da câmara 0
    length: float                 # comprimento do tampo (lado maior), m
    width: float                  # largura do tampo, m
    normal: np.ndarray            # normal do tampo (para cima), no referencial da câmara 0


@dataclass
class Segmentation:
    """Resultado da segmentação, tudo no referencial da MESA."""
    points: np.ndarray        # M x 3: pontos dos objetos (filtrados e reduzidos)
    colors: np.ndarray        # M x 3: cor RGB de cada ponto, em [0, 1]
    clusters: np.ndarray      # M: índice do cluster euclidiano de cada ponto
    boxes_per_cluster: dict   # cluster -> quantas caixas se pensa estarem nesse cluster
    color_groups: dict        # cluster -> rótulo de cor de cada ponto (ou None se a cor não ajuda)


# #####################################################################################################
# 1. DETETAR O PLANO DA MESA COM RANSAC E DEFINIR O REFERENCIAL DA MESA
# #####################################################################################################
# Como na Parte 08 (Ex. 4): primeiro o CHÃO (o maior plano), depois a MESA (RANSAC só nos pontos entre
# 0.5 e 1.0 m acima do chão); o tampo é o maior grupo (DBSCAN) do plano da mesa, e o referencial da mesa
# tem a origem no centro do tampo e o eixo z perpendicular à mesa. O RANSAC é guiado pelas normais
# (Parte 07, Ex. 8): só corre sobre pontos de superfícies horizontais.
# O que não está nas aulas:
#   - o sentido "para cima" (para o lado das câmaras) e o recurso para quando o chão não aparece vêm do
#     RANSAC no centro das imagens (table_plane_guess, table_points_from_center): as câmaras olham
#     para a mesa;
#   - o plano é reajustado por mínimos quadrados (SVD) e o eixo x fica ao longo do lado maior do tampo
#     (retângulo de área mínima, cv2.minAreaRect), a convenção de comum.py. Na Parte 08 o eixo x vai
#     para um ponto qualquer do tampo.
def central_points(views, T_camera0_cameras, fraction=0.4):
    """Pontos 3D (referencial da câmara 0) da zona central de cada imagem.

    Cada câmara aponta para a mesa (é para isso que lá está), por isso os pixels do centro da imagem
    são quase todos do tampo ou das caixas. Assim não é preciso saber quantas câmaras há nem como
    estão dispostas (volta completa, arco, ou uma só). 'fraction' = lado da zona central (40%).
    """
    pts = []
    for view, T in zip(views, T_camera0_cameras):
        h, w = view.xyz_image.shape[:2]
        r0, r1 = int(h * (0.5 - fraction / 2)), int(h * (0.5 + fraction / 2))
        c0, c1 = int(w * (0.5 - fraction / 2)), int(w * (0.5 + fraction / 2))
        p = view.xyz_image[r0:r1, c0:c1].reshape(-1, 3)
        p = p[p[:, 2] > 0]                                # só pixels com medição
        pts.append(transform_points(T, p))
    return np.concatenate(pts)


def table_plane_guess(views, T_camera0_cameras):
    """Primeira estimativa do plano da mesa: RANSAC sobre os pontos centrais das imagens.

    1. Encontra até 3 planos nos pontos centrais (RANSAC repetido, como na Parte 07 Ex. 3: cada
       plano encontrado é retirado e procura-se o seguinte nos pontos que sobram);
    2. A normal de cada plano é orientada para o lado das câmaras ('cima' = de onde a mesa é vista);
    3. Altura de referência: o pixel MESMO no centro de cada imagem (tampo ou topo de uma caixa);
    4. Dos planos paralelos ao maior, com pelo menos 25% dos seus pontos e não mais de 35 cm abaixo
       da referência, fica o MAIS BAIXO: os topos das caixas também são horizontais mas estão acima
       da mesa; o chão também pode aparecer, mas está ~75 cm abaixo.
    Saída: (normal para cima, um ponto do tampo).
    """
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(central_points(views, T_camera0_cameras)))
    rest = pc.voxel_down_sample(0.01)
    centers = np.array([T[:3, 3] for T in T_camera0_cameras])     # posição de cada câmara
    planes = []
    for _ in range(3):
        if len(rest.points) < 300:
            break
        model, inliers = rest.segment_plane(distance_threshold=PLANE_DISTANCE, ransac_n=3, num_iterations=2000)
        normal = np.array(model[:3]) / np.linalg.norm(model[:3])
        point = np.asarray(rest.points)[inliers].mean(axis=0)
        if np.mean((centers - point) @ normal) < 0:               # orienta a normal para as câmaras
            normal = -normal
        planes.append((len(inliers), normal, point))
        rest = rest.select_by_index(inliers, invert=True)         # retira o plano e continua
    if not planes:
        raise RuntimeError("Não foi encontrado nenhum plano no centro das imagens")
    biggest_count, up, _ = max(planes, key=lambda p: p[0])

    # Altura (ao longo de 'up') dos pixels do centro de cada imagem
    center_heights = []
    for v, T in zip(views, T_camera0_cameras):
        h, w = v.xyz_image.shape[:2]
        patch = v.xyz_image[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5].reshape(-1, 3)
        patch = patch[patch[:, 2] > 0]
        center_heights.append(transform_points(T, patch) @ up)
    center_heights = np.concatenate(center_heights)
    reference = np.median(center_heights) if len(center_heights) else -np.inf

    parallel = [p for p in planes if p[0] >= 0.25 * biggest_count and abs(p[1] @ up) > np.cos(np.radians(10))
                and p[2] @ up > reference - 0.35]
    if not parallel:
        parallel = planes
    _, _, point = min(parallel, key=lambda p: p[2] @ up)          # o mais baixo
    return up, point


def horizontal_points(pc, up):
    """Pontos de superfícies HORIZONTAIS, escolhidos pela normal (Parte 07, Ex. 5 e 6).

    1. Calcula a normal de cada ponto (plano ajustado aos vizinhos, raio 3 cm);
    2. Ângulo entre a normal n e a vertical 'up':  theta = arccos(|n . up|)  (|.| porque a normal pode
       apontar para cima ou para baixo, as duas são a mesma superfície);
    3. Fica com os pontos com theta < 10°.
    Saída: nova nuvem só com esses pontos.
    """
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    cos_angle = np.abs(np.asarray(pc.normals) @ up)
    keep = np.where(cos_angle > np.cos(np.radians(HORIZONTAL_MAX_ANGLE)))[0]
    return pc.select_by_index(keep)


def table_points_from_floor(cloud, up, central, verbose=True):
    """Pontos do tampo, pela lógica das aulas: primeiro o CHÃO, depois a mesa (Parte 08, Ex. 4).

    O RANSAC é guiado pelas normais (Parte 07, Ex. 8): só corre sobre pontos de superfícies
    horizontais, para não encontrar 'planos' que cortam paredes, móveis ou caixas na diagonal.
      1. Nuvem agregada (voxel de 1 cm) -> só os pontos com normal vertical (horizontal_points);
      2. Chão = o MAIOR plano desses pontos (segment_plane). A altura de cada ponto acima do chão é
         a distância ao plano, com a normal orientada para cima (para o lado das câmaras);
      3. RANSAC outra vez, só nos pontos horizontais com altura entre 0.5 e 1.0 m -> plano da mesa;
      4. No mesmo plano podem aparecer outras superfícies à mesma altura (um aparador, uma cómoda).
         Clustering euclidiano (DBSCAN) dos pontos do plano e fica o MAIOR grupo, como na aula; mas só
         entre os grupos que aparecem no centro das imagens (as câmaras estão a olhar para a mesa).
    Saída: pontos do tampo (N x 3) ou None se não houver chão ou mesa visíveis (por exemplo, com
    poucas câmaras muito perto da mesa o chão quase não aparece); nesse caso usa-se o outro método.
    """
    pc = horizontal_points(cloud.voxel_down_sample(0.01), up)
    if len(pc.points) < MIN_PLANE_POINTS:
        return None

    # 2. Chão: o maior plano horizontal
    model, inliers = pc.segment_plane(distance_threshold=PLANE_DISTANCE, ransac_n=3, num_iterations=2000)
    if len(inliers) < MIN_PLANE_POINTS:
        return None
    floor_normal = np.array(model[:3]) / np.linalg.norm(model[:3])
    if floor_normal @ up < 0:                                     # normal do chão a apontar para cima
        floor_normal = -floor_normal
    floor_point = np.asarray(pc.points)[inliers].mean(axis=0)
    points = np.asarray(pc.points)
    height = (points - floor_point) @ floor_normal                # altura acima do chão

    # 3. Plano da mesa: RANSAC só nos pontos entre 0.5 e 1.0 m acima do chão
    band = np.where((height > TABLE_HEIGHT_MIN) & (height < TABLE_HEIGHT_MAX))[0]
    if len(band) < MIN_PLANE_POINTS:
        return None
    band_pc = pc.select_by_index(band)
    _, inliers = band_pc.segment_plane(distance_threshold=PLANE_DISTANCE, ransac_n=3, num_iterations=2000)
    plane_points = np.asarray(band_pc.points)[inliers]

    # 4. Maior grupo do plano, entre os que aparecem no centro das imagens
    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plane_points))
                      .cluster_dbscan(eps=0.025, min_points=3))
    near_center = cKDTree(central).query(plane_points, distance_upper_bound=0.02)[0] < np.inf
    candidates = [k for k in range(labels.max() + 1) if np.any(near_center[labels == k])]
    if not candidates:
        return None
    best = max(candidates, key=lambda k: np.sum(labels == k))
    table_points = plane_points[labels == best]
    if len(table_points) < MIN_PLANE_POINTS:
        return None
    if verbose:
        print(f"  chão: {len(inliers)} pontos; mesa a {np.median((table_points - floor_point) @ floor_normal):.2f} m"
              f" do chão")
    return table_points


def table_points_from_center(cloud, up, seed):
    """Pontos do tampo, a partir do plano estimado no centro das imagens (método de recurso).

    Usado quando o chão não é visível. Pontos da nuvem a menos de 1.5 cm do plano estimado e a menos
    de 1 m do ponto estimado; dentro desses, a componente ligada (DBSCAN) que contém o ponto estimado.
    """
    points = np.asarray(cloud.voxel_down_sample(0.01).points)
    rel = points - seed
    height = rel @ up                                             # altura acima do plano
    horizontal = np.linalg.norm(rel - np.outer(height, up), axis=1)   # distância 'no plano'
    table_points = points[(np.abs(height) < 0.015) & (horizontal < 1.0)]

    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(table_points))
                      .cluster_dbscan(eps=0.025, min_points=3))
    if labels.max() >= 0:
        valid = labels >= 0
        nearest = np.argmin(np.linalg.norm(table_points[valid] - seed, axis=1))
        table_points = table_points[labels == labels[valid][nearest]]
    return table_points


def detect_table(cloud, T_camera0_cameras, views, verbose=True):
    """Deteta o tampo da mesa e define o referencial da mesa.

    1. Direção 'para cima' (normal da mesa, aproximada) pelo RANSAC no centro das imagens
       (table_plane_guess);
    2. Pontos do tampo como nas aulas: chão primeiro, depois a mesa entre 0.5 e 1.0 m acima do chão,
       com o RANSAC guiado pelas normais (table_points_from_floor). Se o chão não aparecer, usa o plano
       estimado no centro das imagens (table_points_from_center);
    3. Reajuste do plano por mínimos quadrados (SVD) com os pontos do tampo;
    4. Retângulo de área mínima do tampo visto de cima: dá o centro (origem), a orientação (eixo x
       ao longo do lado maior) e as dimensões do tampo.
    Saída: objeto Table.
    Também é usada pelos extras da Parte 1 (view_tables) com a nuvem de UMA só câmara.
    """
    up, seed = table_plane_guess(views, T_camera0_cameras)

    # 2. Pontos do tampo
    table_points = table_points_from_floor(cloud, up, central_points(views, T_camera0_cameras), verbose)
    if table_points is None:
        if verbose:
            print("  chão não encontrado: mesa pelo plano do centro das imagens")
        table_points = table_points_from_center(cloud, up, seed)

    # Reajuste do plano por mínimos quadrados (SVD): a normal é a direção de menor variância
    centroid = table_points.mean(axis=0)
    _, _, Vt = np.linalg.svd(table_points - centroid)
    normal = Vt[2] if Vt[2] @ up > 0 else -Vt[2]

    # 4. Base ortonormada (e1, e2) no plano e retângulo de área mínima dos pontos do tampo
    e1 = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(normal, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    uv = np.stack([(table_points - centroid) @ e1, (table_points - centroid) @ e2], axis=1)
    (cu, cv), (size_a, size_b), angle = cv2.minAreaRect(uv.astype(np.float32))
    angle = np.radians(angle)
    axis_a = np.cos(angle) * e1 + np.sin(angle) * e2              # direção do lado 'size_a'
    if size_b > size_a:                                            # o eixo x fica ao longo do lado maior
        axis_a = np.cross(normal, axis_a)
        size_a, size_b = size_b, size_a
    x_axis = axis_a / np.linalg.norm(axis_a)
    y_axis = np.cross(normal, x_axis)
    origin = centroid + cu * e1 + cv * e2                          # centro do tampo

    T_camera0_table = make_T(np.column_stack([x_axis, y_axis, normal]), origin)
    if verbose:
        print(f"Parte 2: mesa detetada com {len(table_points)} pontos, tampo de {size_a:.3f} x {size_b:.3f} m")
    return Table(T_camera0_table, float(size_a), float(size_b), normal)


# #####################################################################################################
# 2. RETIRAR A MESA E O FUNDO
# #####################################################################################################
# Como na Parte 08 (Ex. 4.3 e 4.4): a nuvem passa para o referencial da mesa e ficam os pontos acima da
# mesa e dentro dos limites do tampo (só a altura não chega: o chão à volta e outros móveis também
# podem estar a essa altura).
def extract_objects(cloud, table, voxel=OBJECT_VOXEL):
    """Pontos dos objetos sobre a mesa, no referencial da MESA.

    1. Transforma a nuvem para o referencial da mesa: a coordenada z passa a ser a altura acima do tampo;
    2. Fica com os pontos entre 8 mm e 60 cm acima da mesa E dentro dos limites do tampo
       (só a altura não chega: o chão à volta e outros móveis também podem estar a essa altura);
    3. Downsampling (voxel de 4 mm) e remoção de outliers (flying pixels a 'flutuar').
    Saída: (pontos M x 3, cores M x 3).
    """
    T_table_camera0 = np.linalg.inv(table.T_camera0_table)
    points = transform_points(T_table_camera0, np.asarray(cloud.points))
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mask = ((z > ABOVE_TABLE_MIN) & (z < ABOVE_TABLE_MAX) &
            (np.abs(x) < table.length / 2 - TABLE_MARGIN) & (np.abs(y) < table.width / 2 - TABLE_MARGIN))

    objects = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[mask]))
    objects.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[mask])
    objects = objects.voxel_down_sample(voxel)
    objects, _ = objects.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.asarray(objects.points), np.asarray(objects.colors)


# #####################################################################################################
# 3. SEPARAR OS OBJETOS COM CLUSTERING EUCLIDIANO
# #####################################################################################################
# Como na Parte 07 (Ex. 4) e na Parte 08 (Ex. 4.5): cluster_dbscan do Open3D; os pontos soltos (rótulo
# -1) e os clusters pequenos ficam de fora.
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


# ----------------------------------------------------------------------------------------------------
# Figuras da Parte 2 (as janelas são as de visualizacao.py): o resultado esperado do enunciado (objetos
# isolados, resto da nuvem a cinzento) e uma cor por cluster (Parte 08, Ex. 4.5)
# ----------------------------------------------------------------------------------------------------
def table_outline(table):
    """Contorno do tampo (referencial da mesa, z = 0), para desenhar."""
    c = box_corners_2d(0.0, 0.0, 0.0, table.length, table.width)
    return np.column_stack([c, np.zeros(len(c))]), [[0, 1], [1, 2], [2, 3], [3, 0]]


def table_view(T_camera0_camera0, table):
    """Ponto de vista das janelas no referencial da mesa (z para cima, como no simulador): atrás da câmara
    000, a olhar para o centro da mesa."""
    T_table_camera0 = np.linalg.inv(table.T_camera0_table)
    return camera_view(T_table_camera0 @ T_camera0_camera0, np.zeros(3), np.array([0.0, 0.0, 1.0]))


def figure_data(r1, r2):
    """Dados das janelas da Parte 2 (r1 = Parte 1, r2 = resultado de run_segmentation), no referencial da
    mesa: a nuvem agregada inteira (com a cor real e marcando os pontos que são dos objetos) e os pontos
    dos objetos gravados em objetos.ply, e o cluster euclidiano de cada ponto dos objetos."""
    table = r2["table"]
    cloud = transform_points(np.linalg.inv(table.T_camera0_table), np.asarray(r2["cloud"].points))
    # pontos da nuvem agregada que são dos objetos (a nuvem dos objetos tem menos pontos: voxel)
    distance, _ = cKDTree(r2["object_points"]).query(cloud, k=1, distance_upper_bound=2 * OBJECT_VOXEL)
    view = table_view(r1["T_cams"][0], table)
    corners, _ = table_outline(table)
    return {"parte": "parte2_segmentacao", "cena": Path(r2["scene_folder"]).name,
            "points": cloud.astype(np.float32), "colors": colors_uint8(r2["cloud"].colors),
            "is_object": np.isfinite(distance), "objects": r2["object_points"].astype(np.float32),
            "object_colors": colors_uint8(r2["object_colors"]), "table_corners": corners,
            "num_boxes": r2["num_boxes"], "seg_points": r2["seg"].points.astype(np.float32),
            "seg_clusters": r2["seg"].clusters.astype(np.int32), **{f"view_{k}": v for k, v in view.items()}}


def make_windows(data):
    """Janelas da Parte 2:
    1. como a figura do enunciado: na nuvem agregada, os pontos dos objetos com a sua cor e o resto
       (mesa, chão, sala) a cinzento;
    2. cada cluster euclidiano com uma cor diferente, sobre o contorno do tampo (Parte 08, Ex. 4.5);
    3. como antes (show_boxes_cloud): a nuvem de pontos só das caixas, com os eixos do referencial da mesa;
    4. a nuvem de pontos só das caixas (todas juntas, sem cenário) em 3D, vista do lado da câmara 000."""
    colors = np.full((len(data["points"]), 3), 200, np.uint8)            # resto da cena a cinzento
    colors[data["is_object"]] = data["colors"][data["is_object"]]
    outline = lines_item(data["table_corners"], [[0, 1], [1, 2], [2, 3], [3, 0]], (0.2, 0.2, 0.2), 2.0)
    title = f"Parte 2 - {data['cena']}"
    return [{"name": "_objetos", "title": f"{title}: pontos dos objetos isolados (resto da cena a cinzento)",
             "items": [points_item(data["points"], colors), outline], "view": data_view(data)},
            {"name": "_clusters",
             "title": f"{title}: {int(data['seg_clusters'].max()) + 1} clusters euclidianos, uma cor por cluster",
             "items": [points_item(data["seg_points"], CAMERA_COLORS[data["seg_clusters"] % 10], 3.0), outline],
             "view": close_view(data_view(data), data["objects"])},
            {"name": "_nuvem_caixas",
             "title": f"{title}: nuvem de pontos só das caixas ({int(data['num_boxes'])} caixas, referencial da mesa)",
             "items": [points_item(data["objects"], data["object_colors"], 3.0), frame_item(0.1)]},
            {"name": "_nuvem_caixas_3d",
             "title": f"{title}: nuvem de pontos só das caixas em 3D ({int(data['num_boxes'])} caixas, sem cenário)",
             "items": [points_item(data["objects"], data["object_colors"], 3.0)],
             "view": close_view(data_view(data), data["objects"])}]


# Janela 3D do Open3D (opção --open3d)
def show_boxes_cloud(points, colors):
    """A nuvem de pontos só das caixas (objetos.ply), com as cores reais."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.colors = o3d.utility.Vector3dVector(colors)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)    # referencial da mesa
    o3d.visualization.draw_geometries([pc, frame], window_name="Nuvem de pontos só das caixas (referencial da mesa)")


# #####################################################################################################
# 4. USAR A COR DOS PONTOS (Nível 4: caixas em contacto de cores diferentes)
# #####################################################################################################
# Dentro de um cluster com várias caixas, separam-se as caixas pela cor. A cor é descrita pela
# cromaticidade, a mesma ideia do HSV da Parte 02 (Ex. 2b): separar a cor do brilho, para as faces em
# sombra terem a mesma cor que o topo. Não está nas aulas: o k-means (cv2.kmeans, tutorial do OpenCV);
# o inRange da Parte 02 precisa de limites de cor conhecidos, e as cores das caixas não se conhecem.
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


def split_by_color(colors, n):
    """Separa as cores de um cluster com n caixas estimadas (rótulo de cor de cada ponto, ou None).

    Primeiro em n cores; se não der (há caixas com a mesma cor), em n-1, ..., 2 cores: por exemplo, duas
    caixas verdes e uma rosa por cima dão 2 grupos de cor, e o grupo verde é depois dividido pela
    geometria (Parte 3).
    """
    for k in range(n, 1, -1):
        labels = color_split(colors, k)
        if labels is not None:
            return labels
    return None


# #####################################################################################################
# 5. EXTRA: NÚMERO DE CAIXAS DESCONHECIDO (não faz parte das 4 subpartes)
# #####################################################################################################
# O número de caixas NÃO é dado às partes (só o avaliador o conhece): estima-se quantas caixas tem cada
# cluster de 4 formas, e fica a maior (estimate_boxes):
#   1. topos a alturas diferentes (caixas empilhadas ou de alturas diferentes);
#   2. cores diferentes (Nível 4);
#   3. desvio lateral: caixas da mesma altura e cor encostadas em fila, desviadas para o lado ou de larguras
#      diferentes, têm um degrau no contorno do topo em cada junta;
#   4. rotações: caixas rodadas entre si têm paredes viradas para direções diferentes (normais, Parte 07).
# Caixas iguais e PERFEITAMENTE alinhadas continuam a ser uma só (não há nenhuma junta visível; é um dos
# casos impossíveis do Nível 3). A divisão de cada cluster em caixas é feita na Parte 3 (initial_boxes).
# O que não está nas aulas: o perfil do topo (degraus), o histograma circular das orientações das paredes
# (módulo 90°) e o PCA (SVD); as normais e o ângulo com z são os da Parte 07 (Ex. 5 a 7).
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


def count_colors(colors, max_k=4, min_fraction=0.10):
    """Número de cores diferentes num cluster (1 a max_k): o maior k em que color_split separa as cores.

    Para CONTAR caixas, cada cor tem de ter pelo menos 10% dos pontos (mais exigente do que os 5% de
    color_split): numa caixa sozinha, um pequeno grupo de pontos de outra tonalidade (reflexos, bordas
    misturadas com o fundo) não pode passar a contar como mais uma caixa.
    """
    for k in range(max_k, 1, -1):
        labels = color_split(colors, k)
        if labels is not None and np.bincount(labels, minlength=k).min() / len(labels) >= min_fraction:
            return k
    return 1


def wall_orientations(points, min_rotation=MIN_ROTATION, min_fraction=MIN_WALL_FRACTION, min_points=40):
    """Orientações (graus, entre 0 e 90) das paredes das caixas de um cluster, da mais vista para a menos.

    1. Normal de cada ponto (PCA dos vizinhos num raio de 1.5 cm, Parte 07, Ex. 5);
    2. Pontos de PAREDE: normal quase horizontal, isto é, com um ângulo com z perto de 90° (Parte 07,
       Ex. 6 e 7: |n_z| < 0.3);
    3. Direção de cada normal no plano da mesa, MÓDULO 90°: as 4 paredes de uma caixa dão o mesmo valor
       (a caixa com yaw 20° tem paredes viradas para 20°, 110°, 200° e 290°, todas 20° módulo 90°);
    4. Histograma circular de 1° (0° e 90° são vizinhos), suavizado em ±5°; cada máximo local com pelo
       menos 10% das paredes é uma orientação, desde que esteja a mais de MIN_ROTATION graus das já
       aceites e que entre os dois haja um vale (abaixo de metade do pico): o ruído das normais junto às
       arestas alarga cada pico, mas não cria um segundo pico separado por um vale.
    Saída: lista de orientações (graus), vazia se o cluster tiver poucas paredes.
    """
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
    normals = np.asarray(pc.normals)
    wall = np.abs(normals[:, 2]) < WALL_MAX_NZ
    if wall.sum() < min_points:
        return []
    angles = np.degrees(np.arctan2(normals[wall, 1], normals[wall, 0])) % 90.0
    hist = np.bincount(np.floor(angles).astype(int) % 90, minlength=90).astype(float)
    smoothed = np.convolve(np.concatenate([hist[-5:], hist, hist[:5]]), np.ones(11), "valid")   # circular
    is_max = (smoothed >= np.roll(smoothed, 1)) & (smoothed > np.roll(smoothed, -1))
    peaks = []
    for i in np.argsort(-smoothed):
        if smoothed[i] < max(min_fraction * hist.sum(), min_points):
            break
        if not is_max[i]:
            continue
        separate = True
        for j in peaks:
            gap = min(abs(i - j), 90 - abs(i - j))                 # distância circular
            direction = 1 if (j - i) % 90 <= 45 else -1            # caminho mais curto de i para j
            valley = smoothed[[(i + direction * s) % 90 for s in range(1, gap)]]
            if gap < min_rotation or (len(valley) and valley.min() > 0.5 * smoothed[i]):
                separate = False
                break
        if separate:
            peaks.append(int(i))
    return [float(p) for p in peaks]


def count_lateral_steps(points, orientation=None, slice_size=0.01, window=3):
    """Número de caixas em fila num cluster com um só topo, pelos degraus do contorno do topo.

    Caixas da mesma altura e cor encostadas formam um topo plano único. Se estiverem desviadas para o lado
    umas das outras, ou tiverem larguras diferentes, o contorno desse topo tem um degrau em cada junta:
      1. Pontos do topo (top_surface_points), no referencial das caixas: o eixo u é a orientação das
         paredes (wall_orientations) ou a perpendicular, a que for ao longo da fila (o lado maior). Assim
         o contorno de cada caixa é paralelo a u (sem orientação: eixo principal do topo, PCA);
      2. Fatias de 1 cm ao longo de u; em cada uma, os limites do topo em v (percentis 3 e 97, para
         ignorar pontos soltos), e daí o CENTRO e a LARGURA do topo nessa fatia;
      3. Em cada posição compara-se a mediana das 3 fatias antes com a das 3 depois: há uma junta se o
         centro salta mais do que LATERAL_STEP (desvio lateral) ou a largura mais do que WIDTH_STEP
         (larguras diferentes);
      4. Fica o maior salto de cada zona: as juntas ficam a pelo menos MIN_BOX_SIDE umas das outras e
         das pontas da fila.
    Saída: 1 + número de juntas. Caixas iguais perfeitamente alinhadas não têm degrau: contam como uma.
    """
    top = points[top_surface_points(points)]
    if len(top) < 100:
        return 1
    xy = top[:, :2]
    if orientation is None:
        _, _, Vt = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)
        a = np.arctan2(Vt[0, 1], Vt[0, 0])
    else:
        a = np.radians(orientation)
    axes = [np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])]
    u_axis = axes[int(np.argmax([np.ptp(xy @ ax) for ax in axes]))]     # ao longo da fila
    u, v = xy @ u_axis, xy @ np.array([-u_axis[1], u_axis[0]])

    edges = np.arange(u.min(), u.max() + slice_size, slice_size)
    which = np.digitize(u, edges) - 1                                    # fatia de cada ponto
    lo = np.full(len(edges), np.nan)
    hi = np.full(len(edges), np.nan)
    for b in range(len(edges)):
        vb = v[which == b]
        if len(vb) >= 10:
            lo[b], hi[b] = np.percentile(vb, 3), np.percentile(vb, 97)
    center, width = (lo + hi) / 2, hi - lo
    position = edges + slice_size / 2 - u.min()

    score = np.zeros(len(edges))                                         # salto / limiar (> 1: junta)
    for b in range(len(edges)):
        if position[b] < MIN_BOX_SIDE or position[-1] - position[b] < MIN_BOX_SIDE:
            continue
        before, after = slice(max(0, b - window), b), slice(b + 1, b + 1 + window)
        if np.sum(~np.isnan(center[before])) < 2 or np.sum(~np.isnan(center[after])) < 2:
            continue
        jump_center = abs(np.nanmedian(center[after]) - np.nanmedian(center[before]))
        jump_width = abs(np.nanmedian(width[after]) - np.nanmedian(width[before]))
        score[b] = max(jump_center / LATERAL_STEP, jump_width / WIDTH_STEP)
    joints = []
    for b in np.argsort(-score):                                         # do maior salto para o menor
        if score[b] <= 1.0:
            break
        if all(abs(position[b] - position[j]) >= MIN_BOX_SIDE for j in joints):
            joints.append(b)
    return 1 + len(joints)


def estimate_boxes(points, colors, clusters, use_color=True):
    """ESTIMA quantas caixas há em cada cluster (o número de caixas não é dado).

    Em cada cluster euclidiano, o número de caixas é o maior destes quatro:
    - o número de TOPOS a alturas diferentes (count_top_levels): caixas empilhadas, ou encostadas
      com alturas diferentes (Nível 3);
    - o número de CORES diferentes (count_colors): caixas encostadas de cores diferentes (Nível 4);
    - o número de caixas em fila pelos DEGRAUS do contorno do topo (count_lateral_steps): caixas da
      mesma altura e cor desviadas para o lado ou de larguras diferentes. Só em clusters com um só
      topo e uma só orientação das paredes: numa pilha, a caixa de cima (mais estreita) criava degraus
      falsos no topo do cluster, e uma caixa rodada tem o contorno inclinado (degraus em todas as fatias);
    - o número de ORIENTAÇÕES diferentes das paredes (wall_orientations): caixas rodadas entre si.
    Limitação: caixas iguais, da mesma cor e PERFEITAMENTE alinhadas, encostadas umas às outras, ficam
    contadas como uma só (vistas de cima são uma única superfície retangular, sem nenhuma junta; é um
    dos casos impossíveis do Nível 3, como diz o enunciado).
    Todos os clusters contam (os pontos soltos já ficaram de fora no euclidean_clusters).
    Saída: dicionário cluster -> número de caixas.
    """
    counts = {}
    for k in range(clusters.max() + 1):
        mine = clusters == k
        levels = count_top_levels(points[mine])
        n = levels
        if use_color:
            n = max(n, count_colors(colors[mine]))
        orientations = wall_orientations(points[mine])
        n = max(n, len(orientations))
        if levels == 1 and len(orientations) <= 1:
            n = max(n, count_lateral_steps(points[mine], orientations[0] if orientations else None))
        counts[k] = n
    return counts


# #####################################################################################################
# 6. PROGRAMA PRINCIPAL: junta as subpartes por ordem e grava objetos.ply, mesa.json e segmentacao.npz
# #####################################################################################################
OBJECTS_CLOUD = "objetos.ply"           # pontos dos objetos (sem a mesa e o fundo), referencial da mesa
TABLE_FILE = "mesa.json"                # pose e dimensões da mesa
SEGMENTATION_FILE = "segmentacao.npz"   # cluster e grupo de cor de cada ponto


def segment(points, colors, use_color=True, verbose=True):
    """Subpartes 3 e 4 (e o extra): clusters euclidianos -> caixas por cluster (estimadas) -> cor.

    Entrada: os pontos das caixas (objetos.ply, no referencial da mesa, já sem a
    mesa e o fundo) e as suas cores. O número de caixas NÃO é dado: é estimado (estimate_boxes).
    Saída: objeto Segmentation (tudo no referencial da mesa).
    """
    clusters = euclidean_clusters(points)                                    # 3. clustering
    counts = estimate_boxes(points, colors, clusters, use_color=use_color)   # 5. extra: n.º de caixas

    # Pontos soltos (fora dos clusters) ficam de fora
    keep = np.isin(clusters, list(counts.keys()))
    points, colors, clusters = points[keep], colors[keep], clusters[keep]

    # 4. Em cada cluster com várias caixas, tenta separar as caixas pela cor (split_by_color)
    color_groups = {k: split_by_color(colors[clusters == k], n) if (use_color and n > 1) else None
                    for k, n in counts.items()}
    if verbose:
        print(f"  {len(points)} pontos de objetos, {len(counts)} clusters euclidianos; "
              f"caixas estimadas por cluster: {counts} ({sum(counts.values())} caixas); "
              f"clusters separados por cor: {[k for k, g in color_groups.items() if g is not None]}")
    return Segmentation(points, colors, clusters, counts, color_groups)


def save_objects(folder, table, points, colors):
    """Grava a nuvem de pontos só dos objetos (objetos.ply, com cor) e a mesa (mesa.json)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(folder / OBJECTS_CLOUD), cloud)
    info = {"T_camera0_table": np.round(table.T_camera0_table, 9).tolist(),
            "comprimento": table.length, "largura": table.width,
            "normal": np.round(table.normal, 9).tolist()}
    (folder / TABLE_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_objects(folder):
    """Lê a mesa (mesa.json) e a nuvem dos objetos (objetos.ply). Saída: (Table, pontos, cores)."""
    folder = Path(folder)
    info = json.loads((folder / TABLE_FILE).read_text(encoding="utf-8"))
    table = Table(np.array(info["T_camera0_table"]), info["comprimento"], info["largura"], np.array(info["normal"]))
    cloud = o3d.io.read_point_cloud(str(folder / OBJECTS_CLOUD))
    return table, np.asarray(cloud.points), np.asarray(cloud.colors)


def save_segmentation(folder, seg):
    """Grava a segmentação (segmentacao.npz): pontos, cores, cluster de cada ponto, n.º de caixas de cada
    cluster e grupo de cor de cada ponto (-1 nos clusters em que a cor não separou as caixas)."""
    color_group = np.full(len(seg.points), -1)
    for k, groups in seg.color_groups.items():
        if groups is not None:
            color_group[seg.clusters == k] = groups
    np.savez_compressed(Path(folder) / SEGMENTATION_FILE, points=seg.points, colors=seg.colors,
                        clusters=seg.clusters, cluster_ids=np.array(list(seg.boxes_per_cluster.keys())),
                        boxes_per_cluster=np.array(list(seg.boxes_per_cluster.values())),
                        color_group=color_group)


def load_segmentation(folder):
    """Lê o segmentacao.npz e reconstrói o objeto Segmentation."""
    d = np.load(Path(folder) / SEGMENTATION_FILE)
    counts = {int(k): int(n) for k, n in zip(d["cluster_ids"], d["boxes_per_cluster"])}
    color_groups = {}
    for k in counts:
        groups = d["color_group"][d["clusters"] == k]
        color_groups[k] = groups if len(groups) and groups.min() >= 0 else None
    return Segmentation(d["points"], d["colors"], d["clusters"], counts, color_groups)


def run_segmentation(r1, cloud=None, use_color=True, verbose=True):
    """Parte 2 numa cena: mesa, pontos dos objetos e separação das caixas (o número de caixas é estimado).

    Entrada: o resultado da Parte 1 (run_registration ou load_registration) e, se já existir, a nuvem
    agregada (senão as nuvens voltam a ser juntadas com as poses do registo).
    Grava objetos.ply, mesa.json e segmentacao.npz na pasta das partes.
    Saída: dicionário com a mesa, os pontos dos objetos, a segmentação e a nuvem agregada.
    """
    t0 = time.time()
    scene = r1["scene"] if r1.get("scene") is not None else load_scene(r1["scene_folder"])
    registered = r1["registered"]
    if cloud is None:
        cloud = r1.get("cloud")
    if cloud is None:
        cloud, _ = aggregate(scene, r1["T_cams"], registered)
    folder = Path(r1["folder"])

    # 1. Mesa e referencial da mesa (só com as câmaras que entraram na nuvem) e 2. pontos dos objetos
    used = [k for k, ok in enumerate(registered) if ok]
    table = detect_table(cloud, [r1["T_cams"][k] for k in used], [scene.views[k] for k in used], verbose=verbose)
    points, colors = extract_objects(cloud, table)
    # A nuvem dos objetos é gravada e a separação das caixas usa a nuvem gravada (a mesma que a Parte 3
    # e o avaliador leem quando correm sozinhos)
    save_objects(folder, table, points, colors)
    table, points, colors = load_objects(folder)
    if verbose:
        print(f"  {len(points)} pontos dos objetos gravados em {folder / OBJECTS_CLOUD}")

    # 3. clusters euclidianos, 4. cor e 5. (extra) número de caixas estimado
    seg = segment(points, colors, use_color=use_color, verbose=verbose)
    num_boxes = sum(seg.boxes_per_cluster.values())
    save_segmentation(folder, seg)
    if verbose:
        print(f"  segmentação gravada em {folder / SEGMENTATION_FILE} ({time.time() - t0:.1f} s)")
    return {"table": table, "object_points": points, "object_colors": colors, "seg": seg,
            "num_boxes": num_boxes, "cloud": cloud, "scene_folder": Path(r1["scene_folder"])}


def main():
    """Lê os argumentos da linha de comandos e corre a Parte 2."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="pasta da cena (já com a Parte 1 feita)")
    parser.add_argument("--no_color", action="store_true", help="não usar a cor para separar as caixas")
    parser.add_argument("--visualize", action="store_true", help="janelas 3D: objetos isolados e nuvem só das caixas")
    parser.add_argument("--open3d", action="store_true", help="janela antiga do Open3D com a nuvem dos objetos")
    args = parser.parse_args()

    r1 = load_registration(Path(args.scene) / OUTPUT_FOLDER)
    r2 = run_segmentation(r1, use_color=not args.no_color)
    print(f"\nSegmentação gravada em {r1['folder']}. A seguir:\n"
          f"    python savi/parte3_ajuste.py --scene {args.scene}")
    if args.visualize:
        show_windows(make_windows(figure_data(r1, r2)))
    if args.open3d:
        show_boxes_cloud(r2["object_points"], r2["object_colors"])


if __name__ == "__main__":
    main()
