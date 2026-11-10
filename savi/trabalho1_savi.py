#!/usr/bin/env python3
"""SAVIBoxes - Trabalho 1 de SAVI: pose e dimensões de caixas sobre uma mesa, a partir de várias imagens RGB-D.

=====================================================================================================
COMO CORRER
=====================================================================================================
    (a partir da pasta principal do trabalho)
    python savi/trabalho1_savi.py --scene scenes_offline/level_1 --num_boxes 1
    python savi/trabalho1_savi.py --scene test/cena_1 --num_boxes 2 --output results_test/cena_1/result.json

Opções extra (secções 6 a 8 deste ficheiro; não mudam o result.json):
    --evaluate     compara com o ground_truth.json (se a cena o tiver) e mostra tabelas de erros
    --visualize    abre janelas 3D com cada etapa (registo, segmentação, otimização, resultado)
    --figures      grava as figuras top_view.png e optimization.png ao lado do result.json

Avaliação em várias cenas (tabelas do Read_me em results_offline/results.md):
    python savi/trabalho1_savi.py --evaluate_scenes scenes_offline/level_* scenes_offline/stack_*
    python savi/trabalho1_savi.py --evaluate_scenes scenes_offline/level_* --gt_registration

O programa escreve o ficheiro result.json (por omissão dentro da pasta da cena) com a pose de cada
câmara e a pose e as dimensões de cada caixa, tudo no referencial da câmara 000.

=====================================================================================================
ORGANIZAÇÃO DESTE FICHEIRO (pela ordem do enunciado)
=====================================================================================================
    0. Geometria          funções de matrizes 4x4, rotações, caixas
    1. Leitura da cena    ler as imagens / nuvens de pontos e escrever o result.json
    2. PARTE 1 - Registo  estimar a pose de cada câmara em relação à câmara 000
                          (features ASIFT + RANSAC + ICP + hipóteses da mesa + grafo de poses)
    3. PARTE 2 - Segmentação  detetar a mesa, ficar só com os objetos, separá-los (DBSCAN e cor)
    4. PARTE 3 - Ajuste   estimativa inicial + otimização com scipy.optimize.least_squares
    5. Método base        bounding box orientada (só para comparar com a otimização)
    6. Avaliação          erros de uma cena em relação ao ground truth (câmaras, segmentação, caixas, IoU 3D)
    7. Visualização       janelas 3D (Open3D) e figuras PNG (matplotlib)
    8. Várias cenas       corre várias cenas e escreve as tabelas de erros (--evaluate_scenes)
    9. Programa principal junta tudo: run() e main()

=====================================================================================================
CONVENÇÕES
=====================================================================================================
- T_A_B é a matriz 4x4 que leva pontos do referencial B para o referencial A:   p_A = T_A_B @ p_B
- Referencial da MESA: origem no centro do tampo, eixo z perpendicular à mesa e a apontar para cima,
  eixo x ao longo do lado maior do tampo.
- Uma caixa é descrita no referencial da mesa por 7 números:
        [x, y, yaw, length, width, height, z]
  x, y   = centro da caixa visto de cima
  yaw    = rotação em torno de z (radianos)
  length, width, height = dimensões (m)
  z      = altura da BASE da caixa acima da mesa (0 se assenta na mesa; o topo da caixa de baixo se
           está empilhada). O centro da caixa fica à altura z + height/2.
  (z vem em último lugar para os índices dos outros parâmetros serem sempre os mesmos:
   p[2] é sempre o yaw, p[3:6] são sempre as dimensões.)
"""

import argparse
import copy
import glob
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2                                   # OpenCV: features das imagens, k-means, retângulos
import matplotlib
matplotlib.use("Agg")                        # o matplotlib só grava ficheiros PNG (não abre janelas)
import matplotlib.pyplot as plt              # figuras (secção 7)
import numpy as np
import open3d as o3d                         # nuvens de pontos: ICP, RANSAC de planos, DBSCAN
from scipy.optimize import least_squares, linear_sum_assignment   # otimização (Parte 3) / emparelhar caixas (avaliação)
from scipy.spatial import cKDTree            # procura rápida do vizinho mais próximo

reg = o3d.pipelines.registration             # atalho para o módulo de registo do Open3D


# #####################################################################################################
# 0. GEOMETRIA
# #####################################################################################################
# Pequenas funções de matemática usadas em todas as partes.

def make_T(R=np.eye(3), t=np.zeros(3)):
    """Junta uma rotação e uma translação numa matriz homogénea 4x4.

    Entrada: R (3x3, rotação) e t (3, translação).
    Saída:   T (4x4) = [[R, t], [0 0 0 1]].
    """
    T = np.eye(4)
    T[:3, :3] = R          # canto superior esquerdo: rotação
    T[:3, 3] = t           # última coluna: translação
    return T


def transform_points(T, points):
    """Aplica uma transformação 4x4 a muitos pontos de uma vez.

    Entrada: T (4x4) e points (N x 3).
    Saída:   os N pontos transformados, p' = R p + t.
    (Escrito como points @ R^T + t para fazer todos os pontos numa só operação do numpy.)
    """
    return points @ T[:3, :3].T + T[:3, 3]


def rot_z(yaw):
    """Matriz de rotação 3x3 de um ângulo 'yaw' (radianos) em torno do eixo z."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rz4(theta):
    """O mesmo que rot_z, mas já como matriz 4x4 (sem translação)."""
    return make_T(rot_z(theta))


def rotation_angle_deg(R):
    """Ângulo (em graus) de uma matriz de rotação, como na Parte 08 das aulas.

    Qualquer rotação 3D equivale a rodar um ângulo theta em torno de um eixo, e
        theta = arccos((traço(R) - 1) / 2).
    O np.clip evita erros numéricos (valores ligeiramente acima de 1 dariam NaN no arccos).
    Usa-se para medir a diferença entre duas rotações: rotation_angle_deg(R1^T @ R2).
    """
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def kabsch(A, B):
    """Transformação rígida T que leva os pontos A o mais perto possível dos pontos B.

    Entrada: A e B (N x 3), onde A[i] corresponde a B[i].
    Saída:   T (4x4) que minimiza a soma de |T A[i] - B[i]|^2 (mínimos quadrados).
    Método de Kabsch:
      1. centra os dois conjuntos (tira a média a cada um);
      2. matriz de covariância H = A_centrado^T @ B_centrado;
      3. a SVD de H dá a rotação R; a translação é o que falta para os centros coincidirem.
    """
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    # A correção d garante uma rotação verdadeira (determinante +1) e não uma reflexão (espelho)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cb - R @ ca
    return make_T(R, t)


def wrap_angle(a):
    """Põe um ângulo (radianos) no intervalo [-pi, pi). Ex.: 270° -> -90°."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(R):
    """Yaw (rotação em torno de z) de uma matriz de rotação cujo eixo z aponta para cima."""
    return np.arctan2(R[1, 0], R[0, 0])


def box_T_table(x, y, yaw, height, z=0.0):
    """Pose T_table_box de uma caixa: origem no CENTRO da caixa, eixo z para cima.

    O centro fica à altura z + height/2 (z = altura da base acima da mesa).
    """
    return make_T(rot_z(yaw), np.array([x, y, z + height / 2.0]))


def box_corners_2d(x, y, yaw, length, width):
    """Os 4 cantos do retângulo de uma caixa vista de cima (para desenhar as figuras)."""
    half = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * np.array([length, width]) / 2.0
    c, s = np.cos(yaw), np.sin(yaw)
    return half @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


def box_lines_3d(T, length, width, height):
    """Os 8 cantos e as 12 arestas de uma caixa com pose T (para desenhar como LineSet do Open3D)."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    corners = transform_points(T, signs * np.array([length, width, height]) / 2.0)
    edges = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    return corners, edges


def box_signed_distance(points_box, half_dims):
    """Distância COM SINAL de pontos à superfície de uma caixa centrada na origem.

    Entrada: points_box (N x 3) já escritos no referencial da caixa; half_dims = meias-dimensões.
    Saída:   N valores; > 0 fora da caixa, < 0 dentro, 0 em cima da superfície.
             O valor absoluto é a distância do ponto à superfície.
    É a 'signed distance function' clássica de um paralelepípedo:
        q = |p| - meias_dimensões        (q > 0 nos eixos em que o ponto está fora)
        fora   -> norma da parte positiva de q (distância à face, aresta ou canto mais próximo)
        dentro -> o maior componente de q (negativo: menos a distância à face mais próxima)
    """
    q = np.abs(points_box) - half_dims
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


# #####################################################################################################
# 1. LEITURA DA CENA E ESCRITA DO RESULTADO
# #####################################################################################################
# Formato de uma cena (igual para cenas simuladas e reais):
#     cena/
#         point_clouds/000.ply   nuvem de pontos com cor, no referencial da câmara que a adquiriu
#         rgb/000.png            imagem a cores correspondente
#         intrinsics.json        fx, fy, cx, cy, width, height
#         ground_truth.json      (só existe nas cenas de desenvolvimento; NUNCA é usado para estimar)

@dataclass
class View:
    """Uma imagem RGB-D da cena."""
    name: str                       # nome da imagem, por exemplo "000"
    rgb: np.ndarray                 # imagem a cores (H x W x 3, ordem RGB, uint8)
    cloud: o3d.geometry.PointCloud  # nuvem de pontos com cor, no referencial desta câmara
    xyz_image: np.ndarray           # H x W x 3: o ponto 3D de cada pixel (0 onde não há medição)


@dataclass
class Scene:
    """Uma cena completa."""
    folder: Path                    # pasta da cena
    views: list                     # lista de View, ordenada pelo nome (000, 001, ...)
    K: np.ndarray                   # matriz de parâmetros intrínsecos 3x3 da câmara
    width: int                      # largura das imagens (pixels)
    height: int                     # altura das imagens (pixels)
    ground_truth: dict = field(default=None)   # só para avaliar (None se não existir)


def intrinsics_matrix(intrinsics):
    """Constrói a matriz K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] a partir do intrinsics.json."""
    return np.array([[intrinsics["fx"], 0.0, intrinsics["cx"]],
                     [0.0, intrinsics["fy"], intrinsics["cy"]],
                     [0.0, 0.0, 1.0]])


def cloud_to_xyz_image(cloud, K, width, height):
    """Reconstrói a 'imagem de pontos 3D': para cada pixel, o ponto 3D que lhe corresponde.

    Porquê? As features (Parte 1) são encontradas na imagem a cores, em pixels (u, v). Para as
    usar no registo 3D precisamos do ponto 3D de cada pixel. Cada ponto da nuvem foi criado a partir
    de um pixel da imagem de profundidade; projetando o ponto com a matriz K recuperamos esse pixel:
        u = fx * x / z + cx        v = fy * y / z + cy
    Saída: imagem H x W x 3 com o ponto (x, y, z) em cada pixel, ou (0, 0, 0) se não houver medição.
    """
    points = np.asarray(cloud.points)
    xyz_image = np.zeros((height, width, 3), np.float64)
    z = points[:, 2]
    valid = z > 0                                                  # só pontos à frente da câmara
    u = np.round(K[0, 0] * points[valid, 0] / z[valid] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * points[valid, 1] / z[valid] + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)      # só pixels dentro da imagem
    xyz_image[v[inside], u[inside]] = points[valid][inside]
    return xyz_image


def load_scene(folder):
    """Lê uma cena: todas as nuvens de pontos, as imagens a cores, os intrínsecos e o ground truth.

    Entrada: caminho da pasta da cena.
    Saída:   um objeto Scene.
    """
    folder = Path(folder)
    intrinsics = json.loads((folder / "intrinsics.json").read_text(encoding="utf-8"))
    K = intrinsics_matrix(intrinsics)
    width, height = int(intrinsics["width"]), int(intrinsics["height"])

    views = []
    # As nuvens estão ordenadas pelo nome (000.ply, 001.ply, ...); a 000 é a câmara de referência
    for ply in sorted((folder / "point_clouds").glob("*.ply")):
        name = ply.stem
        cloud = o3d.io.read_point_cloud(str(ply))
        # cv2.imread falha no Windows com caminhos com acentos (ex.: "Trabalho prático"), por isso
        # lemos os bytes do ficheiro com numpy e descodificamos a imagem com cv2.imdecode.
        # O OpenCV devolve a ordem BGR; [..., ::-1] troca para RGB (a mesma ordem das cores da nuvem).
        data = np.fromfile(str(folder / "rgb" / f"{name}.png"), dtype=np.uint8)
        rgb = cv2.imdecode(data, cv2.IMREAD_COLOR)[..., ::-1].copy()
        views.append(View(name, rgb, cloud, cloud_to_xyz_image(cloud, K, width, height)))

    # O ground truth só é lido se existir. O programa NUNCA o usa para estimar; só para avaliar.
    ground_truth = None
    gt_path = folder / "ground_truth.json"
    if gt_path.exists():
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    return Scene(folder, views, K, width, height, ground_truth)


def boxes_to_camera0(params, T_camera0_table):
    """Converte as caixas (7 parâmetros no referencial da mesa) para o formato do result.json.

    Entrada: lista de caixas [x, y, yaw, l, w, h, z] e a pose da mesa T_camera0_table.
    Saída:   lista de dicionários com T_camera0_box (pose do centro da caixa no referencial da
             câmara 000), length, width e height.
             T_camera0_box = T_camera0_table @ T_table_box
    """
    return [{"T_camera0_box": T_camera0_table @ box_T_table(p[0], p[1], p[2], p[5], p[6]),
             "length": p[3], "width": p[4], "height": p[5]} for p in params]


def write_result(path, view_names, T_camera0_cameras, boxes):
    """Escreve o result.json no formato pedido no enunciado.

    Entrada: caminho do ficheiro, nomes das imagens, pose de cada câmara (T_camera0_camera) e as
             caixas no formato de boxes_to_camera0.
    Cria a pasta de saída se ainda não existir.
    """
    result = {
        "cameras": [{"image": name, "T_camera0_camera": np.round(T, 6).tolist()}
                    for name, T in zip(view_names, T_camera0_cameras)],
        "boxes": [{"T_camera0_box": np.round(b["T_camera0_box"], 6).tolist(),
                   "length": round(float(b["length"]), 6),
                   "width": round(float(b["width"]), 6),
                   "height": round(float(b["height"]), 6)} for b in boxes],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# #####################################################################################################
# 2. PARTE 1 - REGISTO DAS NUVENS DE PONTOS
# #####################################################################################################
# Objetivo: a pose T_camera0_camera de cada câmara, e uma nuvem agregada toda no referencial da 000.
#
# Para cada par de câmaras vizinhas (i, i+1) estima-se T_i_(i+1):
#   a) hipóteses a partir das FEATURES das imagens (Parte 04): ASIFT + teste de Lowe -> pares de
#      pontos 3D -> RANSAC com Kabsch (Parte 08, Ex. 3, nota);
#   b) hipóteses a partir da MESA vista por cada câmara sozinha (ajuda muito com poucas câmaras);
#   c) cada hipótese é refinada com ICP ponto-a-plano (Parte 08, Ex. 2) e avaliada;
#   d) hipóteses impossíveis (pontos em 'espaço livre') são descartadas; fica a melhor.
# Depois, encadeiam-se as transformações (T_0_2 = T_0_1 @ T_1_2 ...) e, se as câmaras derem a
# volta à mesa, otimiza-se um grafo de poses com fecho de ciclo (a última câmara com a primeira).
#
# Notas sobre as escolhas (vêm de testes com as cenas do simulador):
#   - SIFT normal encontra muito poucas associações corretas entre câmaras a 45° (a perspetiva muda
#     muito). O ASIFT (Affine-SIFT) simula várias inclinações da imagem antes de calcular os
#     descritores e encontra 3 a 10 vezes mais associações corretas.
#   - Features 3D (FPFH, Parte 08 Ex. 3) falham nesta cena: está cheia de planos (mesa, chão,
#     paredes), que têm todos o mesmo descritor, e o RANSAC alinha os planos errados.
#   - O ruído da profundidade cresce com z^2 (a 4 m são ~2.5 cm). No ICP ignoram-se os pontos a mais
#     de 3 m e usa-se um kernel robusto (Tukey).

MAX_ICP_DEPTH = 3.0                      # m: pontos mais longe são demasiado ruidosos para o ICP
ICP_VOXELS = (0.04, 0.02, 0.01, 0.005)   # escalas do ICP, da mais grosseira para a mais fina (m)

# Critérios para aceitar o registo de um par (ver is_confident e free_space_violation)
MIN_PAIR_SCORE = 0.18   # score >= 0.18: o par é sempre de confiança (pares corretos: 0.26-0.48)
LOW_PAIR_SCORE = 0.05   # entre 0.05 e 0.18: só se for claramente melhor que as alternativas
MARGIN_RATIO = 1.5      # 'claramente melhor' = score >= 1.5 x o score da melhor hipótese DIFERENTE
MAX_VIOLATION = 0.10    # fração máxima de pontos em 'espaço livre' (corretas 0.01-0.055, erradas 0.19-0.40)
MAX_LOOP_ANGLE = 100.0  # graus: câmaras com direções de vista mais afastadas não se sobrepõem


# ----------------------------------------------------------------------------------------------------
# 2.1 Pré-processamento das nuvens
# ----------------------------------------------------------------------------------------------------
def crop_depth(cloud, max_depth=MAX_ICP_DEPTH):
    """Fica só com os pontos a menos de max_depth metros da câmara (o z da câmara é a profundidade)."""
    z = np.asarray(cloud.points)[:, 2]
    return cloud.select_by_index(np.nonzero(z < max_depth)[0])


def preprocess(cloud, voxel):
    """Prepara uma nuvem para o ICP: downsampling, remoção de outliers e normais.

    - voxel_down_sample: substitui os pontos de cada cubo (voxel) pela sua média -> densidade
      uniforme e ICP muito mais rápido (Parte 07, Ex. 4);
    - remove_statistical_outlier: retira pontos com distância média aos vizinhos anormal
      (os 'flying pixels' junto às arestas dos objetos);
    - estimate_normals: o ICP ponto-a-plano precisa da normal de cada ponto (Parte 07, Ex. 5).
    """
    down = cloud.voxel_down_sample(voxel)
    down, _ = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return down


# ----------------------------------------------------------------------------------------------------
# 2.2 Estimativa inicial a partir das features das imagens (Parte 04)
# ----------------------------------------------------------------------------------------------------
def detect_features(view):
    """Deteta keypoints e descritores ASIFT na imagem a cores (só onde há medição 3D).

    Saída: (keypoints, descritores), como o detectAndCompute do OpenCV.
    A máscara evita features em pixels sem profundidade (não dariam pontos 3D).
    """
    detector = cv2.AffineFeature_create(cv2.SIFT_create(nfeatures=0))
    gray = cv2.cvtColor(view.rgb, cv2.COLOR_RGB2GRAY)
    mask = (view.xyz_image[..., 2] > 0).astype(np.uint8) * 255
    return detector.detectAndCompute(gray, mask)


def match_3d_pairs(view_a, feats_a, view_b, feats_b, ratio=0.8):
    """Associa features entre duas imagens e devolve os PARES DE PONTOS 3D correspondentes.

    1. Para cada descritor de B procura os 2 mais parecidos em A (knnMatch, k=2) e aplica o teste
       da razão de Lowe (Parte 04, Ex. 4): o melhor tem de ser claramente melhor do que o segundo,
       senão a associação é ambígua e é descartada;
    2. Para cada associação 2D vai buscar o ponto 3D desse pixel em cada imagem (xyz_image).
    Saída: (pontos em A, pontos em B), ambos N x 3, com pontos_A[i] <-> pontos_B[i].
    """
    (kp_a, des_a), (kp_b, des_b) = feats_a, feats_b
    if des_a is None or des_b is None or len(kp_a) < 3 or len(kp_b) < 3:
        return np.zeros((0, 3)), np.zeros((0, 3))
    height, width = view_a.xyz_image.shape[:2]

    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_b, des_a, k=2)
    points_a, points_b = [], []
    for pair in matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance > ratio * second.distance:      # teste da razão de Lowe
            continue
        ub, vb = np.round(kp_b[best.queryIdx].pt).astype(int)    # pixel do keypoint em B
        ua, va = np.round(kp_a[best.trainIdx].pt).astype(int)    # pixel do keypoint em A
        pa = view_a.xyz_image[min(va, height - 1), min(ua, width - 1)]
        pb = view_b.xyz_image[min(vb, height - 1), min(ub, width - 1)]
        if pa[2] > 0 and pb[2] > 0:                       # os dois pixels têm de ter medição 3D
            points_a.append(pa)
            points_b.append(pb)
    return np.array(points_a).reshape(-1, 3), np.array(points_b).reshape(-1, 3)


def ransac_hypotheses(points_src, points_dst, threshold=0.02, iterations=3000, max_hypotheses=4, seed=0):
    """RANSAC: devolve as melhores hipóteses DIFERENTES de T_dst_src a partir de pares de pontos 3D.

    Em cada iteração escolhe 3 pares, calcula a transformação com Kabsch e conta os pares que ficam
    a menos de 'threshold' metros (inliers). É robusto a associações erradas (Parte 06, Ex. 5).

    Amostragem guiada por consistência: uma transformação rígida preserva distâncias, por isso dois
    pares corretos (a1,b1) e (a2,b2) têm |a1-a2| ~ |b1-b2|. Escolhe-se o 1.º par ao acaso e os outros
    dois entre os pares compatíveis com ele. Com 4% de inliers, o RANSAC normal precisaria de dezenas
    de milhares de iterações; assim bastam poucas centenas.

    Porquê várias hipóteses? Texturas repetidas (tábuas do chão, livros) geram grupos de associações
    erradas mas coerentes entre si; às vezes um modelo errado tem mais inliers do que o certo.
    Guardam-se as melhores hipóteses distintas e quem decide é o ICP + o score.
    Saída: lista de (n.º de inliers, T 4x4), da melhor para a pior.
    """
    rng = np.random.default_rng(seed)
    n = len(points_src)
    if n < 3:
        return []

    # Matriz n x n: o par i é compatível com o par j se preservam a distância entre si
    d_src = np.linalg.norm(points_src[:, None] - points_src[None], axis=-1)
    d_dst = np.linalg.norm(points_dst[:, None] - points_dst[None], axis=-1)
    compatible = (np.abs(d_src - d_dst) < threshold) & (d_src > 0.05)   # > 5 cm: evita amostras degeneradas
    neighbours = [np.nonzero(row)[0] for row in compatible]

    hypotheses = []
    for _ in range(iterations):
        first = rng.integers(n)
        if len(neighbours[first]) < 2:
            continue
        second, third = rng.choice(neighbours[first], 2, replace=False)
        if not compatible[second, third]:
            continue
        sample = [first, second, third]
        T = kabsch(points_src[sample], points_dst[sample])
        inliers = np.linalg.norm(transform_points(T, points_src) - points_dst, axis=1) < threshold
        if inliers.sum() >= 3:
            T = kabsch(points_src[inliers], points_dst[inliers])     # refina com todos os inliers
            hypotheses.append((int(inliers.sum()), T))

    # Ordena por n.º de inliers e fica com as melhores que sejam realmente diferentes (> 5 cm ou > 5°)
    hypotheses.sort(key=lambda h: -h[0])
    distinct = []
    for count, T in hypotheses:
        if all(is_different(T, U) for _, U in distinct):
            distinct.append((count, T))
        if len(distinct) == max_hypotheses:
            break
    return distinct


# ----------------------------------------------------------------------------------------------------
# 2.3 ICP e avaliação de um alinhamento
# ----------------------------------------------------------------------------------------------------
def multiscale_icp(source, target, T_init, voxels=ICP_VOXELS):
    """ICP ponto-a-plano em várias escalas, da grosseira para a fina (Parte 08, Ex. 2).

    Entrada: nuvem fonte, nuvem alvo, estimativa inicial T_alvo_fonte, lista de voxels.
    Em cada escala as nuvens são reduzidas com um voxel diferente e a distância máxima de
    correspondência é 1.5 x o voxel. Começar grosseiro dá uma bacia de convergência larga; acabar
    fino dá precisão (é o que a Parte 08 mostra: o resultado depende do limiar e da estimativa
    inicial). O kernel de Tukey dá peso zero aos resíduos grandes: pontos muito ruidosos ou zonas que
    só uma das câmaras vê não puxam a solução.
    Saída: (T final, resultado do Open3D com fitness/rmse, matriz de informação para o grafo de poses).
    """
    source, target = crop_depth(source), crop_depth(target)
    T = T_init
    for voxel in voxels:
        src, dst = preprocess(source, voxel), preprocess(target, voxel)
        estimation = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=voxel))
        result = reg.registration_icp(src, dst, voxel * 1.5, T, estimation,
                                      reg.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7,
                                                                 max_iteration=60))
        T = result.transformation
    information = reg.get_information_matrix_from_point_clouds(src, dst, voxels[-1] * 1.5, T)
    return T, result, information


def alignment_score(source, target, T, distance=0.02, max_color_diff=0.08):
    """Nota de 0 a 1 de um alinhamento, usando as nuvens COMPLETAS (com o fundo) e a cor.

    Saída: fração de pontos da fonte que, depois de transformados por T, têm um ponto do alvo a
    menos de 'distance' metros E com cor parecida.
    Porquê não usar só o fitness do ICP? A Parte 08 pergunta se o fitness chega para saber se o
    alinhamento está certo: não chega. Um alinhamento errado que faz 'deslizar' a mesa sobre si
    própria tem boa sobreposição local, mas falha no fundo e na cor (põe chão em cima da mesa).
    """
    src = source.voxel_down_sample(0.02)
    dst = target.voxel_down_sample(0.02)
    src.transform(T)
    tree = cKDTree(np.asarray(dst.points))                     # para achar o vizinho mais próximo
    dist, idx = tree.query(np.asarray(src.points), k=1)
    color_diff = np.abs(np.asarray(dst.colors)[idx] - np.asarray(src.colors)).mean(axis=1)
    good = (dist < distance) & (color_diff < max_color_diff)
    return good.mean()


def free_space_violation(view_target, K, T_target_source, cloud_source, margin=0.03):
    """Fração de pontos da fonte que ficam em ESPAÇO LIVRE visto pela câmara alvo.

    Ideia: cada pixel da câmara alvo mede a distância até à primeira superfície no seu raio; o
    espaço entre a câmara e essa superfície está VAZIO. Depois de transformados, os pontos da fonte
    não podem ficar nesse espaço vazio (a câmara alvo tê-los-ia visto à frente).
    Como: projeta-se cada ponto na imagem do alvo (com K) e contam-se os que ficam mais de
    'margin' metros À FRENTE da profundidade medida nesse pixel.
    Um alinhamento correto tem poucos (só ruído nas arestas); um errado põe, por exemplo, a mesa a
    atravessar o chão. Funciona mesmo quando as duas vistas quase não se sobrepõem.
    """
    points = transform_points(T_target_source, np.asarray(cloud_source.voxel_down_sample(0.02).points))
    points = points[points[:, 2] > 0.1]                         # só pontos à frente da câmara alvo
    if len(points) == 0:
        return 0.0
    uvw = points @ K.T                                          # projeção: (u*z, v*z, z)
    u = np.round(uvw[:, 0] / uvw[:, 2]).astype(int)
    v = np.round(uvw[:, 1] / uvw[:, 2]).astype(int)
    h, w = view_target.xyz_image.shape[:2]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    measured = view_target.xyz_image[v[inside], u[inside], 2]  # profundidade medida pela câmara alvo
    has_depth = measured > 0
    if not has_depth.any():
        return 0.0
    z = measured[has_depth]
    # O ruído da profundidade cresce com z^2 (~2.5 cm a 4 m): a margem cresce da mesma forma
    tolerance = margin + 0.005 * z ** 2
    return float(np.mean(points[inside, 2][has_depth] < z - tolerance))


def is_plausible(view_a, view_b, K, T):
    """Uma hipótese T_a_b é plausível se nenhuma das câmaras vir pontos da outra em espaço livre."""
    if K is None:
        return True
    return max(free_space_violation(view_a, K, T, view_b.cloud),
               free_space_violation(view_b, K, np.linalg.inv(T), view_a.cloud)) < MAX_VIOLATION


def is_different(T1, T2, max_translation=0.05, max_angle=5.0):
    """True se duas transformações são soluções diferentes (diferem mais de 5 cm ou mais de 5°)."""
    return (np.linalg.norm(T1[:3, 3] - T2[:3, 3]) > max_translation or
            rotation_angle_deg(T1[:3, :3].T @ T2[:3, :3]) > max_angle)


def is_confident(candidates):
    """Decide se o melhor candidato é de confiança. candidates = lista de (score, T, origem).

    - score alto (>= MIN_PAIR_SCORE): sim;
    - score médio (>= LOW_PAIR_SCORE): só se for pelo menos MARGIN_RATIO vezes melhor do que a
      melhor hipótese que leva a uma solução DIFERENTE. Com poucas câmaras a sobreposição entre
      vistas é pequena e o score do par correto fica baixo, mas o da alternativa errada fica muito
      mais baixo ainda;
    - abaixo disso: não.
    """
    best = max(candidates, key=lambda c: c[0])
    if best[0] >= MIN_PAIR_SCORE:
        return True
    if best[0] < LOW_PAIR_SCORE:
        return False
    rivals = [c[0] for c in candidates if is_different(c[1], best[1])]
    return not rivals or best[0] >= MARGIN_RATIO * max(rivals)


# ----------------------------------------------------------------------------------------------------
# 2.4 Hipóteses a partir da mesa vista por cada câmara
# ----------------------------------------------------------------------------------------------------
def table_hypotheses(table_a, table_b):
    """Hipóteses de T_a_b a partir do tampo da mesa visto por cada câmara SOZINHA.

    Cada câmara vê a mesa inteira, por isso consegue estimar sozinha o referencial da mesa
    (T_camera_table: centro do tampo, normal, eixo do lado maior). Se as duas virem a mesma mesa:
        T_a_b = T_a_table @ Rz(theta) @ inv(T_b_table)
    O retângulo é simétrico, por isso theta = 0° ou 180° (e também ±90° se a mesa for quase
    quadrada). São as caixas que desfazem a ambiguidade, através do alignment_score.
    Com poucas câmaras (45° a 75° entre vizinhas) as features quase não têm associações corretas
    (às vezes 4 em 400), mas estas hipóteses ficam a ~1 cm da solução.
    """
    thetas = [0.0, np.pi]
    if abs(table_a.length - table_a.width) < 0.1:
        thetas += [np.pi / 2, -np.pi / 2]
    return [table_a.T_camera0_table @ rz4(t) @ np.linalg.inv(table_b.T_camera0_table) for t in thetas]


def view_tables(views):
    """A mesa detetada em cada imagem sozinha (None se não for possível).

    Usa a mesma função da Parte 2 (detect_table) com a nuvem de uma só câmara e a identidade como
    pose. Só se aceita se o tampo parecer completo (largura > 30 cm).
    """
    tables = []
    for view in views:
        try:
            table = detect_table(view.cloud, [np.eye(4)], [view], verbose=False)
            tables.append(table if table.width > 0.3 else None)
        except Exception:
            tables.append(None)
    return tables


# ----------------------------------------------------------------------------------------------------
# 2.5 Registo de um par de imagens
# ----------------------------------------------------------------------------------------------------
def feature_candidates(view_a, feats_a, view_b, feats_b, ratio, iterations, max_hypotheses):
    """Hipóteses das features (RANSAC), cada uma refinada com ICP grosseiro e avaliada.

    Saída: (lista de (score, T, origem), n.º de pares 3D encontrados).
    """
    pts_a, pts_b = match_3d_pairs(view_a, feats_a, view_b, feats_b, ratio=ratio)
    hypotheses = ransac_hypotheses(pts_b, pts_a, iterations=iterations, max_hypotheses=max_hypotheses)
    candidates = []
    for inliers, T_init in hypotheses:
        # Só as escalas grosseiras do ICP: chega para escolher a hipótese e é mais rápido
        T, _, _ = multiscale_icp(view_b.cloud, view_a.cloud, T_init, voxels=ICP_VOXELS[:2])
        candidates.append((alignment_score(view_b.cloud, view_a.cloud, T), T, f"features ({inliers} inliers)"))
    return candidates, len(pts_a)


def register_pair(view_a, feats_a, view_b, feats_b, table_a=None, table_b=None, K=None, verbose=True):
    """Estima T_a_b: a transformação que leva pontos da câmara b para a câmara a.

    1. Hipóteses da mesa (table_hypotheses), se as mesas de cada vista forem dadas;
    2. Hipóteses das features: pares 3D ASIFT -> várias hipóteses de RANSAC;
    3. Cada hipótese é refinada com ICP; as que põem pontos em espaço livre são descartadas
       (is_plausible); das restantes fica a de maior alignment_score;
    4. Se o resultado não for de confiança (is_confident), tenta-se outra vez com mais esforço:
       teste de Lowe mais permissivo (mais associações), mais iterações e mais hipóteses;
    5. A vencedora é refinada nas escalas finas do ICP.
    Saída: (T_a_b, matriz de informação, score, de confiança sim/não).
    """
    candidates = []
    if table_a is not None and table_b is not None:
        for T_init in table_hypotheses(table_a, table_b):
            # Já está a ~1 cm: só as escalas finas (as grosseiras podiam fazê-la 'deslizar')
            T, _, _ = multiscale_icp(view_b.cloud, view_a.cloud, T_init, voxels=ICP_VOXELS[2:])
            candidates.append((alignment_score(view_b.cloud, view_a.cloud, T), T, "mesa"))
    found, num_pairs = feature_candidates(view_a, feats_a, view_b, feats_b,
                                          ratio=0.8, iterations=3000, max_hypotheses=4)
    candidates += found
    candidates = [c for c in candidates if is_plausible(view_a, view_b, K, c[1])]

    if not candidates or not is_confident(candidates):
        if verbose:
            best_score = max((c[0] for c in candidates), default=0.0)
            print(f"    {view_a.name} <- {view_b.name}: pouca confiança ({best_score:.3f}), "
                  f"nova tentativa com mais esforço...")
        retry, num_pairs = feature_candidates(view_a, feats_a, view_b, feats_b,
                                              ratio=0.9, iterations=10000, max_hypotheses=10)
        candidates += [c for c in retry if is_plausible(view_a, view_b, K, c[1])]
    if not candidates:
        # Nenhuma hipótese plausível: parte-se da identidade (o par vai ser marcado como falhado)
        T, _, _ = multiscale_icp(view_b.cloud, view_a.cloud, np.eye(4), voxels=ICP_VOXELS[:2])
        candidates = [(0.0, T, "identidade (nenhuma hipótese plausível)")]
    confident = is_confident(candidates)

    # Refinamento final da hipótese vencedora nas escalas finas
    score, T_coarse, origin = max(candidates, key=lambda c: c[0])
    T, result, information = multiscale_icp(view_b.cloud, view_a.cloud, T_coarse, voxels=ICP_VOXELS[2:])
    if verbose:
        flag = "" if confident else "  <-- FALHOU (não será usado)"
        print(f"    {view_a.name} <- {view_b.name}: {num_pairs} pares 3D, {len(candidates)} hipóteses, "
              f"escolhida: {origin} (score={score:.3f}); "
              f"ICP fitness={result.fitness:.3f}, rmse={result.inlier_rmse * 1000:.2f} mm{flag}")
    return T, information, score, confident


# ----------------------------------------------------------------------------------------------------
# 2.6 Registo de todas as imagens
# ----------------------------------------------------------------------------------------------------
def poses_from_edges(n, edges):
    """Pose T_0_k de cada câmara, percorrendo o grafo de arestas a partir da câmara 0.

    Cada aresta (i, j, T_i_j, ...) pode ser percorrida nos dois sentidos:
        T_0_j = T_0_i @ T_i_j        ou        T_0_i = T_0_j @ inv(T_i_j)
    (pesquisa em largura). Câmaras que não se conseguem alcançar ficam com None.
    """
    poses = [None] * n
    poses[0] = np.eye(4)
    queue = [0]
    while queue:
        a = queue.pop(0)
        for i, j, T, *_ in edges:
            if i == a and poses[j] is None:
                poses[j] = poses[i] @ T
                queue.append(j)
            elif j == a and poses[i] is None:
                poses[i] = poses[j] @ np.linalg.inv(T)
                queue.append(i)
    return poses


def register_scene(scene, loop_closure=True, verbose=True):
    """Regista todas as imagens da cena.

    1. Regista cada imagem com a seguinte (i -> i+1) e encadeia: T_0_(i+1) = T_0_i @ T_i_(i+1).
       O erro de cada par acumula-se ao longo da cadeia (deriva);
    2. Fecho de ciclo: regista também a última imagem com a primeira, mas só se as duas olharem
       para lados parecidos (< MAX_LOOP_ANGLE). Com poucas câmaras num arco a primeira e a última
       olham em direções opostas e não têm nada em comum;
    3. Pares sem confiança NÃO são usados (um par errado estragaria todas as câmaras seguintes).
       Numa volta completa cada câmara continua alcançável pelo outro lado. Se uma câmara ficar
       isolada, tenta-se registá-la com uma câmara a 2 posições de distância;
    4. Grafo de poses (Open3D): distribui o erro pelas arestas de forma ótima.
    Saída: (T_camera0_camera de cada câmara depois do grafo, T_camera0_camera só por encadeamento).
    """
    views = scene.views
    n = len(views)
    if verbose:
        print("Parte 1: registo")
        print("  a detetar features ASIFT e a mesa em cada imagem...")
    features = [detect_features(v) for v in views]
    tables = view_tables(views)

    def pair(a, b):
        """Atalho: regista a imagem b com a imagem a."""
        return register_pair(views[a], features[a], views[b], features[b], tables[a], tables[b],
                             scene.K, verbose)

    # 1. Pares consecutivos (odometria). Cada aresta: (i, j, T_i_j, informação, confiança, incerta)
    edges = []
    for i in range(n - 1):
        T, info, _, ok = pair(i, i + 1)
        edges.append((i, i + 1, T, info, ok, False))

    # Encadeamento simples (só para comparação no relatório)
    T_chain = [np.eye(4)]
    for i, j, T, *_ in edges:
        T_chain.append(T_chain[i] @ T)

    if not loop_closure or n < 3:
        return T_chain, T_chain

    # 2. Fecho de ciclo, só se a primeira e a última câmara olharem para lados parecidos
    good = [e for e in edges if e[4]]                 # só as arestas de confiança
    poses = poses_from_edges(n, good)
    if poses[n - 1] is not None:
        # Ângulo entre os eixos óticos (eixo z) da câmara 0 e da última câmara
        angle = np.degrees(np.arccos(np.clip(poses[n - 1][2, 2], -1.0, 1.0)))
    else:
        angle = 0.0                                   # não se sabe: tenta-se o fecho na mesma
    if angle < MAX_LOOP_ANGLE:
        T, info, _, ok = pair(n - 1, 0)
        if ok:
            good.append((n - 1, 0, T, info, ok, True))
            poses = poses_from_edges(n, good)
    elif verbose:
        print(f"  sem fecho de ciclo: a câmara {views[n - 1].name} olha {angle:.0f}° ao lado da câmara "
              f"{views[0].name} (câmaras num arco)")

    # 3. Câmaras isoladas: tenta-se registá-las com câmaras já posicionadas a 2 posições de distância
    for k in range(n):
        if poses[k] is not None:
            continue
        for m in sorted({(k - 2) % n, (k + 2) % n} - {k}):
            if poses[m] is None:
                continue
            if verbose:
                print(f"  câmara {views[k].name} isolada: a registar com {views[m].name}")
            T, info, _, ok = pair(m, k)
            if ok:
                good.append((m, k, T, info, ok, True))
                poses = poses_from_edges(n, good)
                break
    # Último recurso: câmaras ainda sem pose ficam com a pose do encadeamento
    poses = [p if p is not None else T_chain[k] for k, p in enumerate(poses)]

    # 4. Grafo de poses: cada nó é uma câmara (pose T_0_i), cada aresta é o registo de um par
    graph = reg.PoseGraph()
    for T_node in poses:
        graph.nodes.append(reg.PoseGraphNode(T_node))
    for i, j, T, info, _, uncertain in good:
        # O Open3D quer a aresta (source=j, target=i) com T a levar pontos de 'source' para 'target'
        graph.edges.append(reg.PoseGraphEdge(j, i, T, info, uncertain=uncertain))
    option = reg.GlobalOptimizationOption(max_correspondence_distance=ICP_VOXELS[-1] * 1.5,
                                          edge_prune_threshold=0.25, reference_node=0)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    reg.global_optimization(graph, reg.GlobalOptimizationLevenbergMarquardt(),
                            reg.GlobalOptimizationConvergenceCriteria(), option)
    T_graph = [np.array(node.pose) for node in graph.nodes]

    # A câmara 0 tem de ser exatamente a identidade (é a referência)
    T0_inv = np.linalg.inv(T_graph[0])
    return [T0_inv @ T for T in T_graph], T_chain


def aggregate(scene, T_camera0_cameras):
    """Junta todas as nuvens (resolução original) no referencial da câmara 0.

    Saída: (nuvem agregada, índice da câmara que adquiriu cada ponto).
    """
    aggregated = o3d.geometry.PointCloud()
    camera_index = []
    for k, (view, T) in enumerate(zip(scene.views, T_camera0_cameras)):
        cloud = copy.deepcopy(view.cloud)       # cópia, para não alterar a nuvem original
        cloud.transform(T)                      # leva os pontos para o referencial da câmara 0
        aggregated += cloud
        camera_index.append(np.full(len(cloud.points), k))
    return aggregated, np.concatenate(camera_index)


# #####################################################################################################
# 3. PARTE 2 - SEGMENTAÇÃO
# #####################################################################################################
# Sobre a nuvem AGREGADA:
#   1. detetar o plano da mesa (RANSAC, Parte 07) e definir o referencial da mesa (Parte 08, Ex. 4);
#   2. retirar a mesa e o fundo: ficam os pontos acima da mesa e dentro dos limites do tampo;
#   3. separar os objetos com clustering euclidiano (DBSCAN);
#   4. dentro de um cluster com várias caixas, tentar separá-las pela cor (Nível 4).

PLANE_DISTANCE = 0.008       # m: distância máxima ao plano para um ponto ser do plano (RANSAC)
ABOVE_TABLE_MIN = 0.008      # m: pontos abaixo disto (ruído do tampo) são retirados
ABOVE_TABLE_MAX = 0.60       # m: pontos acima disto já não são objetos sobre a mesa
TABLE_MARGIN = 0.01          # m: margem para dentro dos limites do tampo
CLUSTER_EPS = 0.015          # m: no DBSCAN, pontos a menos de 1.5 cm ficam no mesmo cluster
OBJECT_VOXEL = 0.004         # m: voxel usado nos pontos dos objetos


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


# ----------------------------------------------------------------------------------------------------
# 3.1 Plano da mesa e referencial da mesa
# ----------------------------------------------------------------------------------------------------
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


HORIZONTAL_MAX_ANGLE = 10    # graus: normal a menos disto da vertical -> ponto de uma superfície horizontal
TABLE_HEIGHT_MIN = 0.5       # m: o tampo está entre 0.5 m e 1.0 m acima do chão (Parte 08, Ex. 4)
TABLE_HEIGHT_MAX = 1.0
MIN_PLANE_POINTS = 300       # n.º mínimo de pontos (voxel de 1 cm) para aceitar um plano do chão ou da mesa


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


# ----------------------------------------------------------------------------------------------------
# 3.2 Retirar a mesa e o fundo
# ----------------------------------------------------------------------------------------------------
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


def segment(cloud, table, num_boxes, use_color=True, verbose=True):
    """Segmentação completa: objetos sobre a mesa -> clusters -> caixas por cluster -> cor.

    Saída: objeto Segmentation (tudo no referencial da mesa).
    """
    points, colors = extract_objects(cloud, table)
    clusters = euclidean_clusters(points)
    counts = allocate_boxes(points, clusters, num_boxes)

    # Pontos de clusters que não foram usados (ruído) ficam de fora
    keep = np.isin(clusters, list(counts.keys()))
    points, colors, clusters = points[keep], colors[keep], clusters[keep]

    # Em cada cluster com várias caixas, tenta separar as caixas pela cor
    color_groups = {k: color_split(colors[clusters == k], n) if (use_color and n > 1) else None
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


def split_profile(group, bin_size=0.005, min_piece=0.04, min_step=0.01):
    """Divide um grupo de pontos do topo em dois, no 'degrau' do contorno.

    Caixas da mesma altura encostadas formam um topo plano único. Se tiverem larguras diferentes (ou
    estiverem desalinhadas), o contorno desse topo tem um degrau onde acaba uma caixa e começa a outra:
      1. eixo principal do topo (PCA em x, y): u ao longo da fila de caixas, v na perpendicular;
      2. fatias de 5 mm ao longo de u; em cada fatia, os limites do topo em v (percentis 10 e 90);
      3. o degrau é onde estes limites mais mudam (compara a fatia 1 cm antes com a 1 cm depois);
      4. corta-se aí, desde que fiquem pelo menos 4 cm de cada lado.
    Saída: os dois pedaços, ou None se não houver nenhum degrau claro (> 1 cm).
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
    3. Se houver menos níveis do que caixas, divide o grupo com maior área até haver n:
         - forma de L e faltam pelo menos 2 caixas: corta pelo canto interior do L (split_corner);
         - method="kmeans":  k-means no plano (divide a área ao meio, sem olhar ao contorno);
         - method="profile": no degrau do contorno (split_profile); sem degrau, usa o k-means.
       A divisão é só uma aproximação: a otimização corrige a fronteira. Mas, se começar muito
       longe, pode ficar presa num mínimo local, por isso run() experimenta os dois métodos;
    4. Retângulo de cada caixa visto de cima. Para um nível que não foi dividido e que tem caixas
       mais altas no cluster, o topo visível pode ser só uma moldura ou um pedaço (o resto está
       tapado pela caixa de cima). Juntam-se então as PAREDES dessa caixa: pontos entre o nível de
       baixo e o seu topo, fora dos retângulos das caixas mais altas (que já estão calculados).
    Saída: lista de n caixas [x, y, yaw, length, width, topo] (ainda sem z: ver place_on_supports).
    """
    top = points[top_surface_points(points)]
    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(top)).cluster_dbscan(0.012, 5))
    pieces = [top[labels == k] for k in range(labels.max() + 1) if np.sum(labels == k) >= 30]

    # 2. Níveis: pedaços ordenados pela altura; um novo nível começa quando a altura salta > 1.5 cm
    pieces.sort(key=lambda g: np.median(g[:, 2]))
    levels = []
    for g in pieces:
        if levels and np.median(g[:, 2]) - np.median(levels[-1][:, 2]) < 0.015:
            levels[-1] = np.vstack([levels[-1], g])
        else:
            levels.append(g)
    levels.sort(key=len, reverse=True)
    groups = [(g, True) for g in levels[:n]] if levels else [(top, False)]   # (pontos, nível inteiro?)

    # 3. Divisões até haver n grupos
    while len(groups) < n:
        areas = [rect_params(g[:, :2])[3] * rect_params(g[:, :2])[4] for g, _ in groups]
        g, _ = groups.pop(int(np.argmax(areas)))               # o grupo com maior área
        split = split_corner(g) if n - len(groups) >= 3 else None
        if split is None and method == "profile":
            split = split_profile(g)
        if split is None:
            split = split_kmeans(g)
        groups += [(p, False) for p in split]

    # 4. Retângulos, do grupo mais alto para o mais baixo
    groups.sort(key=lambda item: -np.median(item[0][:, 2]))
    heights = [np.median(g[:, 2]) for g, _ in groups]
    boxes = []
    for i, (g, whole_level) in enumerate(groups):
        rect = rect_params(g[:, :2])
        if whole_level and i > 0:                               # há caixas mais altas no cluster
            below = max([h for h in heights if h < heights[i] - 0.015], default=0.0)
            wall = points[(points[:, 2] > below + 0.01) & (points[:, 2] < heights[i] - 0.01)]
            for b in boxes:                                     # fora das caixas mais altas
                wall = wall[~inside_rect(wall[:, :2], b[:5], margin=0.01)]
            candidates = np.vstack([g, wall])
            rect = rect_params(candidates[largest_component(candidates), :2])
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
    return any(n > 1 and seg.color_groups.get(k) is None for k, n in seg.boxes_per_cluster.items())


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
            for g in range(n):
                sel = np.nonzero(groups == g)[0]
                sel = sel[largest_component(pts[sel])]          # retira pontos soltos com a cor errada
                params.append(np.append(rect_params(pts[sel, :2]), np.percentile(pts[sel, 2], 98)))
                colors.append(np.median(chroma_features(cols[sel]), axis=0))
                labels[idx[sel]] = first + g
            params[first:] = place_on_supports(params[first:])
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
    """Mostra no terminal as tabelas de erros de uma cena (r = resultado de trabalho1_savi.run)."""
    gt = r["scene"].ground_truth
    print("\nRegisto (erro de cada câmara):")
    print(f"  {'imagem':8s}{'transl. (mm)':>14s}{'rot. (°)':>10s}")
    for name, t, a in camera_errors(r["T_cams"], gt):
        print(f"  {name:8s}{t:14.2f}{a:10.3f}")

    n_sets, n_gt, seg_rows = evaluate_segmentation(
        r["seg"].points, r["box_labels"], r["table"].T_camera0_table, np.asarray(r["cloud"].points), gt)
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


def show_registration(scene, T_est, camera_index_cloud=None, T_gt=None):
    """Nuvem agregada: cores reais | cor por câmara | (se houver GT) estimada a vermelho sobre a verdadeira a verde."""
    from .registration import aggregate
    cloud, cam = aggregate(scene, T_est)
    cloud = cloud.voxel_down_sample(0.005)
    o3d.draw_geometries([cloud], window_name="Nuvem agregada (cores reais)")

    by_camera = o3d.geometry.PointCloud()
    for k, (view, T) in enumerate(zip(scene.views, T_est)):
        c = copy.deepcopy(view.cloud).voxel_down_sample(0.005)
        c.transform(T)
        c.paint_uniform_color(CAMERA_COLORS[k % 10])
        by_camera += c
    o3d.draw_geometries([by_camera], window_name="Nuvem agregada (cor = câmara)")

    if T_gt is not None:
        est, _ = aggregate(scene, T_est)
        gt, _ = aggregate(scene, T_gt)
        est = est.voxel_down_sample(0.005)
        gt = gt.voxel_down_sample(0.005)
        est.paint_uniform_color((0.9, 0.1, 0.1))
        gt.paint_uniform_color((0.1, 0.7, 0.2))
        o3d.draw_geometries([gt, est], window_name="Estimada (vermelho) vs verdadeira (verde)")


def show_segmentation(seg, labels):
    """Pontos dos objetos coloridos pela caixa a que foram atribuídos."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seg.points))
    pc.colors = o3d.utility.Vector3dVector([BOX_COLORS[l % len(BOX_COLORS)] if l >= 0 else (0.5, 0.5, 0.5)
                                            for l in labels])
    o3d.draw_geometries([pc], window_name="Segmentação (cor = caixa)")


def animate_optimization(points, colors, problem, delay=0.15):
    """Mostra a evolução das caixas ao longo da otimização, sobre a nuvem de pontos."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.Visualizer()
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
    o3d.draw_geometries(geoms, window_name="Caixas finais")


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
        r = run(folder, num_boxes, use_color=use_color, gt_registration=gt_registration)
        gt = r["scene"].ground_truth

        if not gt_registration:
            cams = camera_errors(r["T_cams"], gt)
            chain = camera_errors(r["T_chain"], gt)
            t, a = np.array([c[1] for c in cams]), np.array([c[2] for c in cams])
            reg_lines.append(f"| {name} | {t[1:].mean():.2f} | {t.max():.2f} | {a[1:].mean():.3f} | {a.max():.3f} | "
                             f"{max(c[1] for c in chain):.2f} |")

        n_sets, n_gt, seg_rows = evaluate_segmentation(
            r["seg"].points, r["box_labels"], r["table"].T_camera0_table, np.asarray(r["cloud"].points), gt)
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
def run(scene_folder, num_boxes, use_color=True, loop_closure=True, gt_registration=False, verbose=True):
    """Corre o sistema completo numa cena e devolve um dicionário com os resultados de cada etapa.

    gt_registration=True usa as poses VERDADEIRAS das câmaras em vez do registo. Só serve para
    avaliar a segmentação e o ajuste isoladamente (como pede o enunciado); nunca no resultado final.
    """
    t0 = time.time()
    scene = load_scene(scene_folder)
    if verbose:
        print(f"Cena {scene.folder.name}: {len(scene.views)} imagens, {num_boxes} caixas")

    # ---------------- Parte 1: registo ----------------
    if gt_registration:
        T_cams = [np.array(c["T_camera0_camera"]) for c in scene.ground_truth["cameras"]]
        T_chain = T_cams
    else:
        T_cams, T_chain = register_scene(scene, loop_closure=loop_closure, verbose=verbose)
    cloud, _ = aggregate(scene, T_cams)

    # ---------------- Parte 2: segmentação ----------------
    table = detect_table(cloud, T_cams, scene.views, verbose=verbose)
    seg = segment(cloud, table, num_boxes, use_color=use_color, verbose=verbose)

    # ---------------- Parte 3: ajuste ----------------
    # Multi-start: com caixas da mesma cor em contacto a estimativa inicial é incerta e a otimização
    # pode ficar presa num mínimo local. Experimentam-se duas estimativas iniciais (k-means e degraus
    # do contorno) e fica a que acaba com MENOR CUSTO (os custos são comparáveis: a função objetivo e
    # os pontos são os mesmos nas duas tentativas).
    methods = ["kmeans", "profile"] if needs_geometric_split(seg) else ["kmeans"]
    # Centros das câmaras no referencial da mesa (para o ajuste saber que faces podem ser vistas)
    T_table_camera0 = np.linalg.inv(table.T_camera0_table)
    camera_centers = np.array([(T_table_camera0 @ T)[:3, 3] for T in T_cams])
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
        print(f"  tempo total: {time.time() - t0:.1f} s")

    return {"scene": scene, "T_cams": T_cams, "T_chain": T_chain, "cloud": cloud, "table": table, "seg": seg,
            "init_params": np.array(init_params), "box_labels": box_labels,
            "params_bbox": params_bbox, "params_opt": params_opt, "problem": problem,
            "fit_points": fit_points, "fit_assoc": fit_assoc}


def main():
    """Lê os argumentos da linha de comandos, corre o sistema e escreve o result.json."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", help="pasta da cena")
    parser.add_argument("--num_boxes", type=int, help="número de caixas na mesa")
    parser.add_argument("--output", help="ficheiro de saída (por omissão: <cena>/result.json)")
    parser.add_argument("--method", choices=["opt", "bbox"], default="opt",
                        help="método usado no result.json: otimização (opt) ou bounding box (bbox)")
    parser.add_argument("--no_color", action="store_true", help="não usar a cor (segmentação nem otimização)")
    parser.add_argument("--no_loop_closure", action="store_true", help="registo só por encadeamento")
    parser.add_argument("--evaluate", action="store_true", help="compara com o ground_truth.json, se existir")
    parser.add_argument("--visualize", action="store_true", help="abre janelas 3D com cada etapa")
    parser.add_argument("--figures", action="store_true", help="grava top_view.png e optimization.png")
    parser.add_argument("--evaluate_scenes", nargs="+", metavar="CENA",
                        help="avalia várias cenas com ground truth (aceita wildcards) e escreve as tabelas")
    parser.add_argument("--gt_registration", action="store_true",
                        help="com --evaluate_scenes: usa as poses verdadeiras (avalia só segmentação e ajuste)")
    args = parser.parse_args()

    # Modo "várias cenas": só avaliação, não escreve result.json
    if args.evaluate_scenes:
        evaluate_scenes(args.evaluate_scenes, gt_registration=args.gt_registration,
                        use_color=not args.no_color, output=args.output)
        return
    if args.scene is None or args.num_boxes is None:
        parser.error("indica --scene e --num_boxes (ou --evaluate_scenes)")

    r = run(args.scene, args.num_boxes, use_color=not args.no_color, loop_closure=not args.no_loop_closure)

    # Escrita do result.json
    params = r["params_opt"] if args.method == "opt" else r["params_bbox"]
    boxes = boxes_to_camera0(params, r["table"].T_camera0_table)
    output = Path(args.output) if args.output else Path(args.scene) / "result.json"
    write_result(output, [v.name for v in r["scene"].views], r["T_cams"], boxes)
    print(f"\nResultado escrito em {output}")

    # Opções extra (secções 6 e 7)
    gt = r["scene"].ground_truth
    if args.evaluate or args.visualize or args.figures:
        gt_params = gt_params_in_table(gt, r["table"].T_camera0_table) if gt else None
        if args.figures:
            save_top_view(output.parent / "top_view.png", r["seg"].points, r["seg"].colors,
                                        r["params_opt"], r["params_bbox"], gt_params, title=Path(args.scene).name)
            save_evolution(output.parent / "optimization.png", r["fit_points"], r["problem"])
        if args.evaluate and gt:
            print_evaluation(r)
        if args.visualize:
            T_gt = [np.array(c["T_camera0_camera"]) for c in gt["cameras"]] if gt else None
            show_registration(r["scene"], r["T_cams"], T_gt=T_gt)
            show_segmentation(r["seg"], r["box_labels"])
            animate_optimization(r["fit_points"], np.full_like(r["fit_points"], 0.6), r["problem"])
            show_final(r["seg"].points, r["seg"].colors, r["params_opt"], r["params_bbox"], gt_params)


if __name__ == "__main__":
    main()
