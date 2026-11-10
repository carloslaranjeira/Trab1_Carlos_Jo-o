"""Parte 2: deteção da mesa e segmentação dos objetos, sobre a nuvem de pontos AGREGADA.

Passos:
    1. Detetar o plano da mesa com RANSAC e definir um referencial da mesa
       (origem no centro do tampo, z perpendicular à mesa e a apontar para cima, x ao longo do comprimento);
    2. Retirar a mesa e o fundo: ficam os pontos acima da mesa e dentro dos seus limites;
    3. Separar os objetos com clustering euclidiano (DBSCAN);
    4. Usar a cor quando for útil: dentro de um cluster com várias caixas, tentar separá-las por cor (Nível 4).
"""

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d

from .geometry import make_T, transform_points


# Parâmetros da segmentação (em metros)
PLANE_DISTANCE = 0.008       # distância máxima ao plano para um ponto ser da mesa (RANSAC)
ABOVE_TABLE_MIN = 0.008      # pontos abaixo disto (ruído da mesa) são retirados
ABOVE_TABLE_MAX = 0.60       # pontos acima disto já não são objetos sobre a mesa
TABLE_MARGIN = 0.01          # margem dentro dos limites da mesa
CLUSTER_EPS = 0.015          # distância do DBSCAN: pontos a menos de 1.5 cm ficam no mesmo cluster
OBJECT_VOXEL = 0.004         # voxel usado nos pontos dos objetos (ajuste e bounding box)


@dataclass
class Table:
    """Mesa detetada: pose no referencial da câmara 0 e dimensões do tampo."""
    T_camera0_table: np.ndarray
    length: float
    width: float
    normal: np.ndarray        # normal do plano no referencial da câmara 0 (para cima)


@dataclass
class Segmentation:
    """Resultado da segmentação, tudo no referencial da MESA."""
    points: np.ndarray        # M x 3: pontos dos objetos (depois de filtrados e reduzidos)
    colors: np.ndarray        # M x 3: cores RGB em [0, 1]
    clusters: np.ndarray      # M: índice do cluster euclidiano de cada ponto
    boxes_per_cluster: dict   # cluster -> número de caixas que se pensa estarem nesse cluster
    color_groups: dict        # cluster -> rótulo de cor de cada ponto do cluster (ou None se a cor não ajuda)


# --------------------------------------------------------------------------------------
# 1. Plano da mesa e referencial da mesa
# --------------------------------------------------------------------------------------
def estimate_up_and_center(T_camera0_cameras):
    """Usa as poses das câmaras para saber, aproximadamente, onde está a mesa e onde é 'cima'.

    As câmaras estão num anel à volta da mesa e olham para baixo, para o centro.
    - O centro do anel fica por cima do centro da mesa;
    - A direção 'cima' é a normal do plano que passa pelos centros das câmaras, com o sinal
      escolhido de forma a que as câmaras olhem para baixo.
    Isto serve só para guiar o RANSAC (escolher entre os vários planos horizontais da sala).
    """
    centers = np.array([T[:3, 3] for T in T_camera0_cameras])
    forwards = np.array([T[:3, 2] for T in T_camera0_cameras])   # eixo z (direção de visão) de cada câmara
    ring_center = centers.mean(axis=0)
    if len(centers) >= 3:
        _, _, Vt = np.linalg.svd(centers - ring_center)
        up = Vt[2]                       # direção de menor variância dos centros = normal do anel
    else:
        up = -T_camera0_cameras[0][:3, 1]  # só uma câmara: 'cima' é aproximadamente -y da câmara
    if np.dot(forwards.mean(axis=0), up) > 0:   # as câmaras olham para baixo
        up = -up
    ring_radius = np.mean(np.linalg.norm(np.cross(centers - ring_center, up), axis=1)) if len(centers) >= 3 else 1.0
    return up / np.linalg.norm(up), ring_center, max(ring_radius, 0.6)


def detect_table(cloud, T_camera0_cameras, verbose=True):
    """Deteta o tampo da mesa com RANSAC e define o referencial da mesa.

    1. Seleciona os pontos dentro do anel de câmaras e abaixo delas (onde tem de estar a mesa);
    2. Aplica RANSAC (segment_plane) várias vezes, retirando os inliers de cada plano encontrado;
    3. Dos planos quase horizontais com muitos pontos, escolhe o mais alto (o chão fica por baixo;
       os topos das caixas são planos pequenos);
    4. Fica com a maior componente ligada dos inliers (DBSCAN) e reajusta o plano por SVD;
    5. Calcula o retângulo de área mínima do tampo visto de cima: dá o centro, a orientação e as
       dimensões da mesa.
    """
    up_guess, ring_center, ring_radius = estimate_up_and_center(T_camera0_cameras)

    down = cloud.voxel_down_sample(0.01)
    points = np.asarray(down.points)
    rel = points - ring_center
    height = rel @ up_guess                                        # altura em relação ao anel de câmaras
    horizontal = np.linalg.norm(rel - np.outer(height, up_guess), axis=1)
    candidates = np.nonzero((horizontal < ring_radius) & (height < -0.1))[0]
    rest = down.select_by_index(candidates)

    # RANSAC repetido: encontra até 6 planos
    planes = []
    for _ in range(6):
        if len(rest.points) < 500:
            break
        model, inliers = rest.segment_plane(distance_threshold=PLANE_DISTANCE, ransac_n=3, num_iterations=2000)
        normal = np.array(model[:3]) / np.linalg.norm(model[:3])
        plane_points = np.asarray(rest.points)[inliers]
        if abs(normal @ up_guess) > np.cos(np.radians(20)):          # plano quase horizontal
            planes.append((len(inliers), (plane_points - ring_center).mean(axis=0) @ up_guess, plane_points))
        rest = rest.select_by_index(inliers, invert=True)

    if not planes:
        raise RuntimeError("Não foi encontrado nenhum plano horizontal para a mesa")
    biggest = max(p[0] for p in planes)
    # O mais alto entre os planos com pelo menos 10% dos pontos do maior.
    # (o chão pode ter mais pontos que a mesa, porque se vê à volta dela; os topos das caixas
    #  têm muito menos de 10% dos pontos, por isso não são confundidos com a mesa)
    _, _, table_points = max((p for p in planes if p[0] >= 0.1 * biggest), key=lambda p: p[1])

    # Maior componente ligada (retira pedaços de outros objetos à mesma altura)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(table_points))
    labels = np.array(pc.cluster_dbscan(eps=0.025, min_points=3))
    if labels.max() >= 0:
        table_points = table_points[labels == np.bincount(labels[labels >= 0]).argmax()]

    # Reajuste do plano por mínimos quadrados (SVD): a normal é a direção de menor variância
    centroid = table_points.mean(axis=0)
    _, _, Vt = np.linalg.svd(table_points - centroid)
    normal = Vt[2] if Vt[2] @ up_guess > 0 else -Vt[2]

    # Base ortonormada no plano da mesa (e1, e2) e retângulo de área mínima dos pontos do tampo
    e1 = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(normal, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    uv = np.stack([(table_points - centroid) @ e1, (table_points - centroid) @ e2], axis=1)
    (cu, cv), (size_a, size_b), angle = cv2.minAreaRect(uv.astype(np.float32))
    angle = np.radians(angle)
    axis_a = np.cos(angle) * e1 + np.sin(angle) * e2              # direção do lado 'size_a'
    if size_b > size_a:                                            # x ao longo do lado maior
        axis_a = np.cross(normal, axis_a)
        size_a, size_b = size_b, size_a
    x_axis = axis_a / np.linalg.norm(axis_a)
    y_axis = np.cross(normal, x_axis)
    origin = centroid + cu * e1 + cv * e2

    T_camera0_table = make_T(np.column_stack([x_axis, y_axis, normal]), origin)
    if verbose:
        print(f"Parte 2: mesa detetada com {len(table_points)} pontos, tampo de {size_a:.3f} x {size_b:.3f} m")
    return Table(T_camera0_table, float(size_a), float(size_b), normal)


# --------------------------------------------------------------------------------------
# 2. Retirar a mesa e o fundo
# --------------------------------------------------------------------------------------
def object_mask(points_table, table):
    """Máscara dos pontos (no referencial da mesa) que estão acima da mesa e dentro dos seus limites."""
    x, y, z = points_table[:, 0], points_table[:, 1], points_table[:, 2]
    return ((z > ABOVE_TABLE_MIN) & (z < ABOVE_TABLE_MAX) &
            (np.abs(x) < table.length / 2 - TABLE_MARGIN) & (np.abs(y) < table.width / 2 - TABLE_MARGIN))


def extract_objects(cloud, table, voxel=OBJECT_VOXEL):
    """Devolve os pontos dos objetos (no referencial da mesa), já reduzidos e sem outliers."""
    T_table_camera0 = np.linalg.inv(table.T_camera0_table)
    points = transform_points(T_table_camera0, np.asarray(cloud.points))
    mask = object_mask(points, table)

    objects = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[mask]))
    objects.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[mask])
    objects = objects.voxel_down_sample(voxel)
    # Remove pontos isolados (flying pixels que ficaram a 'flutuar' entre os objetos e o fundo)
    objects, _ = objects.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.asarray(objects.points), np.asarray(objects.colors)


# --------------------------------------------------------------------------------------
# 3. Clustering euclidiano
# --------------------------------------------------------------------------------------
def euclidean_clusters(points, eps=CLUSTER_EPS, min_points=10, min_cluster_size=150):
    """DBSCAN: agrupa pontos que estão ligados por cadeias de vizinhos a menos de eps metros.

    Clusters muito pequenos são considerados ruído (rótulo -1).
    Os clusters são renumerados por ordem decrescente de tamanho.
    """
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.array(pc.cluster_dbscan(eps=eps, min_points=min_points))
    result = np.full(len(points), -1)
    sizes = [(np.sum(labels == k), k) for k in range(labels.max() + 1)]
    for new_label, (size, k) in enumerate(sorted((s for s in sizes if s[0] >= min_cluster_size), reverse=True)):
        result[labels == k] = new_label
    return result


def footprint_area(points_xy):
    """Área (vista de cima) ocupada por um conjunto de pontos: área do retângulo de área mínima."""
    (_, _), (a, b), _ = cv2.minAreaRect(points_xy.astype(np.float32))
    return a * b


def allocate_boxes(points, clusters, num_boxes):
    """Decide quantas caixas há em cada cluster.

    - Se há tantos clusters como caixas: uma caixa por cluster (Níveis 1 e 2);
    - Se há menos clusters que caixas: algumas caixas tocam-se e formam um só cluster (Níveis 3 e 4).
      As caixas extra são dadas, uma a uma, ao cluster com maior área por caixa;
    - Se há mais clusters que caixas: ficam os maiores (os outros são ruído ou outros objetos).
    """
    labels = [k for k in range(clusters.max() + 1)]
    labels = labels[:num_boxes]
    counts = {k: 1 for k in labels}
    areas = {k: footprint_area(points[clusters == k, :2]) for k in labels}
    for _ in range(num_boxes - len(labels)):
        k = max(labels, key=lambda c: areas[c] / counts[c])
        counts[k] += 1
    return counts


# --------------------------------------------------------------------------------------
# 4. Separação por cor
# --------------------------------------------------------------------------------------
def chroma_features(colors):
    """Converte cores RGB em (a*, b*) do espaço Lab.

    No espaço Lab, L* é a luminosidade e (a*, b*) a cromaticidade. Usar só (a*, b*) torna a
    comparação de cores quase insensível às sombras e à iluminação de cada face da caixa.
    """
    lab = cv2.cvtColor(colors.astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return lab[:, 1:]


def color_split(colors, k, min_separation=4.0):
    """Tenta separar um cluster em k grupos de cor com k-means em (a*, b*).

    Devolve os rótulos de cor de cada ponto, ou None se as cores não forem distinguíveis
    (caixas da mesma cor). O critério é a razão entre a distância entre centros de cor
    e o desvio padrão (robusto) dentro de cada grupo.
    """
    if k < 2:
        return None
    features = chroma_features(colors)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    _, labels, centers = cv2.kmeans(features.astype(np.float32), k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    # Desvio padrão robusto dentro dos grupos (mediana dos desvios absolutos * 1.4826)
    spread = np.median(np.linalg.norm(features - centers[labels], axis=1)) * 1.4826 + 1e-6
    center_dist = min(np.linalg.norm(centers[i] - centers[j]) for i in range(k) for j in range(i + 1, k))
    # Todos os grupos têm de ter uma fração razoável dos pontos
    smallest = np.bincount(labels, minlength=k).min() / len(labels)
    if center_dist / spread < min_separation or smallest < 0.05:
        return None
    return labels


# --------------------------------------------------------------------------------------
# Segmentação completa
# --------------------------------------------------------------------------------------
def segment(cloud, table, num_boxes, use_color=True, verbose=True):
    """Executa os passos 2 a 4 sobre a nuvem agregada (já com a mesa detetada)."""
    points, colors = extract_objects(cloud, table)
    clusters = euclidean_clusters(points)
    counts = allocate_boxes(points, clusters, num_boxes)

    # Pontos de clusters não usados (ruído) ficam de fora
    keep = np.isin(clusters, list(counts.keys()))
    points, colors, clusters = points[keep], colors[keep], clusters[keep]

    color_groups = {}
    for k, n in counts.items():
        color_groups[k] = color_split(colors[clusters == k], n) if (use_color and n > 1) else None

    if verbose:
        print(f"  {len(points)} pontos de objetos, {len(counts)} clusters euclidianos; "
              f"caixas por cluster: {counts}; "
              f"clusters separados por cor: {[k for k, g in color_groups.items() if g is not None]}")
    return Segmentation(points, colors, clusters, counts, color_groups)
