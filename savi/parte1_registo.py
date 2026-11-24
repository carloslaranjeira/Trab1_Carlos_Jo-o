#!/usr/bin/env python3
"""SAVIBoxes - Trabalho 1 de SAVI. PARTE 1: REGISTO (pose de cada câmara e nuvem de pontos agregada).

=====================================================================================================
COMO CORRER
=====================================================================================================
    (a partir da pasta principal do trabalho)
    python savi/parte1_registo.py --scene test/cena_1
    python savi/parte1_registo.py --scene test/cena_1 --visualize   # janela: nuvem agregada (2 subplots)
    python savi/parte1_registo.py --scene test/cena_1 --open3d      # janelas 3D do Open3D (+ ground truth)

Não precisa do número de caixas. Grava <cena>/partes/registo.json com a pose de cada câmara em relação
à câmara 000 (T_camera0_camera) e as câmaras que entraram na nuvem agregada.

Ordem das partes (o run_test.py corre-as todas, cena a cena):
    parte1_registo.py      -> <cena>/partes/registo.json   (pose de cada câmara)
    extra_fotometria.py    -> <cena>/partes/fotometria.json (ganho e desvio de cada imagem)
    parte2_segmentacao.py  -> <cena>/partes/objetos.ply, mesa.json, segmentacao.npz (precisa do n.º de caixas)
    parte3_ajuste.py       -> result.json e <cena>/partes/ajuste.npz
    avaliador.py           -> erros em relação ao ground truth (só nas cenas que o têm)

=====================================================================================================
ORGANIZAÇÃO DESTE FICHEIRO
=====================================================================================================
    0. Geometria          funções de matrizes 4x4, rotações, caixas (usadas também pelas outras partes)
    1. Leitura da cena    ler as imagens / nuvens de pontos e escrever o result.json
    2. Registo            estimar a pose de cada câmara em relação à câmara 000 e juntar as nuvens
                          (features ASIFT + RANSAC + ICP + hipóteses da mesa + grafo de poses)
    3. Saída              gravar e ler o registo.json
    4. Visualização       janela com 2 subplots: nuvem agregada com a cor dos píxeis | com uma cor por
                          câmara (e funções de desenho usadas também pelas outras partes)
    5. Programa principal run_registration() e main()

=====================================================================================================
CONVENÇÕES (usadas em todas as partes)
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
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2                                   # OpenCV: features das imagens, retângulos
import matplotlib.pyplot as plt              # janelas com subplots (secção 4)
import numpy as np
import open3d as o3d                         # nuvens de pontos: ICP, RANSAC de planos, DBSCAN
from scipy.spatial import cKDTree            # procura rápida do vizinho mais próximo

# Correção da exposição (extra_fotometria.py): usada para comparar cores de imagens diferentes no
# registo (alignment_score) e para juntar as nuvens com a mesma exposição (aggregate)
from extra_fotometria import exposure_corrected, exposure_fit


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


def load_ground_truth(scene_folder):
    """O ground_truth.json da cena, ou None se não existir (só para avaliar, NUNCA para estimar)."""
    path = Path(scene_folder) / "ground_truth.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


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
CONFIRMED_RATIO = 0.75  # hipótese confirmada pela mesa E pelas features: aceita se score >= 0.75 x o melhor
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
# 2.3 ICP e avaliação de um alinhamento (a correção da exposição está em extra_fotometria.py)
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
    src_colors = exposure_corrected(np.asarray(src.colors), np.asarray(dst.colors)[idx], dist < distance)
    color_diff = np.abs(np.asarray(dst.colors)[idx] - src_colors).mean(axis=1)
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
      mais baixo ainda. Sem nenhuma hipótese diferente, tem de ser MARGIN_RATIO x LOW_PAIR_SCORE;
    - abaixo disso: não.
    """
    best = max(candidates, key=lambda c: c[0])
    if best[0] >= MIN_PAIR_SCORE:
        return True
    if best[0] < LOW_PAIR_SCORE:
        return False
    # Sem rivais, compara-se com uma rival imaginária no limite (LOW_PAIR_SCORE): uma hipótese sozinha
    # não é prova de que está certa (no teste 10 a única que sobrou estava rodada 180°, score 0.054)
    rivals = [c[0] for c in candidates if is_different(c[1], best[1])]
    return best[0] >= MARGIN_RATIO * max(rivals + [LOW_PAIR_SCORE])


def is_confirmed(candidate, candidates):
    """True se outra hipótese, vinda da OUTRA fonte (mesa vs features), chega à mesma solução.

    As hipóteses da mesa e as das features são independentes: se as duas levam à mesma pose, é muito
    pouco provável que estejam ambas erradas da mesma maneira.
    """
    source = candidate[2] == "mesa"
    return any(c is not candidate and (c[2] == "mesa") != source and not is_different(c[1], candidate[1])
               for c in candidates)


def choose_candidate(candidates):
    """Escolhe a hipótese de um par. Saída: (candidato vencedor, de confiança sim/não).

    1. Se a de maior score é de confiança (is_confident), é essa;
    2. Senão, se houver UMA solução confirmada pela mesa e pelas features (is_confirmed) com score
       >= CONFIRMED_RATIO x o melhor, é essa, e é de confiança. Caso do teste 16, cena 1 (pilha da
       mesma cor, 2 imagens, exposição diferente): a hipótese da mesa rodada 180° tinha score 0.110,
       sozinha; a correta tinha 0.095, mas a mesa e 4 hipóteses das features concordavam com ela.
    3. Senão, a de maior score, sem confiança.
    """
    best = max(candidates, key=lambda c: c[0])
    if is_confident(candidates):
        return best, True
    confirmed = [c for c in candidates if c[0] >= LOW_PAIR_SCORE and is_confirmed(c, candidates)]
    if confirmed:
        top = max(confirmed, key=lambda c: c[0])
        single_solution = all(not is_different(c[1], top[1]) for c in confirmed)
        if single_solution and top[0] >= CONFIRMED_RATIO * best[0]:
            return top, True
    return best, False


# ----------------------------------------------------------------------------------------------------
# 2.4 Hipóteses a partir da mesa vista por cada câmara (a deteção da mesa é a da Parte 2)
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
    from parte2_segmentacao import detect_table     # aqui dentro: a Parte 2 importa este ficheiro
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

    if not candidates or not choose_candidate(candidates)[1]:
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
    (score, T_coarse, origin), confident = choose_candidate(candidates)
    if confident and not is_confident(candidates):
        origin += ", confirmada pela mesa e pelas features"

    # Refinamento final da hipótese vencedora nas escalas finas
    T, result, information = multiscale_icp(view_b.cloud, view_a.cloud, T_coarse, voxels=ICP_VOXELS[2:])
    if verbose:
        flag = "" if confident else "  <-- FALHOU (não será usado)"
        print(f"    {view_a.name} <- {view_b.name}: {num_pairs} pares 3D, {len(candidates)} hipóteses, "
              f"escolhida: {origin} (score={score:.3f}); "
              f"ICP fitness={result.fitness:.3f}, rmse={result.inlier_rmse * 1000:.2f} mm{flag}")
    return T, information, score, confident


# ----------------------------------------------------------------------------------------------------
# 2.6 Registo de todas as imagens e nuvem agregada
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

    good = [e for e in edges if e[4]]                 # só as arestas de confiança
    poses = poses_from_edges(n, good)
    if not loop_closure or n < 3:
        # Uma câmara ligada à 0 por um par falhado fica com a pose do encadeamento (o result.json
        # precisa de uma pose), mas NÃO entra na nuvem: com 2 imagens e o registo errado, a mesma
        # pilha aparecia duas vezes, em dois sítios da mesa (teste 16, cena 1)
        return T_chain, T_chain, [p is not None for p in poses]

    # 2. Fecho de ciclo, só se a primeira e a última câmara olharem para lados parecidos
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
    # Último recurso: câmaras ainda sem pose ficam com a pose do encadeamento (só para o result.json;
    # não entram na nuvem)
    registered = [p is not None for p in poses]
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
    return [T0_inv @ T for T in T_graph], T_chain, registered


def aggregate(scene, T_camera0_cameras, registered=None, photometric=True, exposures=None):
    """Junta as nuvens (resolução original) no referencial da câmara 0, com a exposição de cada
    imagem corrigida para a da câmara 0.

    registered: lista de True/False por câmara; as câmaras com False (registo falhado) ficam de fora.
    photometric=False: sem a correção da exposição (só para a janela do extra_fotometria.py).
    exposures: lista onde se acrescenta (imagem, s, b) de cada câmara (para o extra_fotometria.py).
    Saída: (nuvem agregada, índice da câmara que adquiriu cada ponto).
    """
    aggregated = o3d.geometry.PointCloud()
    camera_index = []
    for k, (view, T) in enumerate(zip(scene.views, T_camera0_cameras)):
        if registered is not None and not registered[k]:
            continue
        cloud = copy.deepcopy(view.cloud)       # cópia, para não alterar a nuvem original
        cloud.transform(T)                      # leva os pontos para o referencial da câmara 0
        s, b = 1.0, 0.0                         # câmara 0 (a referência) ou sem correção
        if len(aggregated.points) > 0 and photometric:
            # Exposição: a cor desta imagem passa a ter a exposição das anteriores. A cromaticidade
            # (chroma_features) cancela o ganho s mas não o desvio b, e com cores de exposições
            # diferentes duas caixas de cores parecidas (laranja e castanho) deixavam de se separar.
            small, ref = cloud.voxel_down_sample(0.01), aggregated.voxel_down_sample(0.01)
            dist, idx = cKDTree(np.asarray(ref.points)).query(np.asarray(small.points), k=1)
            s, b = exposure_fit(np.asarray(small.colors), np.asarray(ref.colors)[idx], dist < 0.01)
            cloud.colors = o3d.utility.Vector3dVector(np.clip(s * np.asarray(cloud.colors) + b, 0.0, 1.0))
        if exposures is not None:
            exposures.append((view.name, s, b))
        aggregated += cloud
        camera_index.append(np.full(len(cloud.points), k))
    return aggregated, np.concatenate(camera_index)


# #####################################################################################################
# 3. SAÍDA: registo.json (é a entrada das outras partes)
# #####################################################################################################
OUTPUT_FOLDER = "partes"              # pasta, dentro da cena, onde cada parte grava o seu resultado
REGISTRATION_FILE = "registo.json"    # pose de cada câmara


def save_registration(folder, scene, T_cams, T_chain, registered):
    """Grava o registo.json: a pose de cada câmara (grafo de poses e só encadeamento) e as câmaras que
    entraram na nuvem agregada (False = registo falhado). É a entrada das outras partes.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    info = {
        "cena": str(Path(scene.folder).resolve()),
        "imagens": [v.name for v in scene.views],
        "T_camera0_camera": [np.round(T, 9).tolist() for T in T_cams],
        "T_camera0_camera_so_encadeamento": [np.round(T, 9).tolist() for T in T_chain],
        "camaras_registadas": [bool(ok) for ok in registered],
    }
    (folder / REGISTRATION_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_registration(folder):
    """Lê o registo.json gravado por save_registration.

    Saída: dicionário com a pasta da cena, os nomes das imagens, a pose de cada câmara (grafo de
    poses e só encadeamento), as câmaras registadas e a pasta das partes. As nuvens não são lidas aqui.
    """
    info = json.loads((Path(folder) / REGISTRATION_FILE).read_text(encoding="utf-8"))
    return {"scene_folder": Path(info["cena"]), "images": info["imagens"],
            "T_cams": [np.array(T) for T in info["T_camera0_camera"]],
            "T_chain": [np.array(T) for T in info["T_camera0_camera_so_encadeamento"]],
            "registered": info["camaras_registadas"], "folder": Path(folder)}


# #####################################################################################################
# 4. VISUALIZAÇÃO
# #####################################################################################################
# Uma cor por câmara (as 10 cores 'tab10' do matplotlib)
CAMERA_COLORS = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
                          [0.58, 0.4, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.5, 0.5, 0.5],
                          [0.74, 0.74, 0.13], [0.09, 0.75, 0.81]])


# As janelas são feitas com o matplotlib (permite subplots). Cada parte tem duas funções:
#   figure_data(...)  -> dicionário só com arrays (pode ser gravado num .npz e desenhado noutro processo:
#                        é assim que o run_test.py abre as janelas sem parar o programa);
#   make_figures(data) -> lista de janelas {"name", "fig"} (e "start" para as animações).
DISPLAY_POINTS = 20000    # pontos desenhados em cada subplot 3D (com mais, o matplotlib fica lento a rodar)
DISPLAY_RADIUS = 2.0      # m: só se desenham os pontos a menos disto do centro das câmaras (mesa e à volta)


def sample_indices(n, max_points, seed=0):
    """Índices de uma amostra aleatória de no máximo max_points pontos (para desenhar)."""
    if n <= max_points:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))


def camera_to_display(points):
    """Pontos no referencial da câmara 000 (x para a direita, y para baixo, z para a frente) -> eixos de
    desenho com o 'cima' para cima: (x, z, -y)."""
    p = np.asarray(points, float).reshape(-1, 3)
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=1)


def scatter_3d(ax, points, colors, size=0.5):
    """Nuvem de pontos num subplot 3D."""
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=size, depthshade=False, linewidths=0)


def set_equal_3d(ax, points):
    """Mesma escala nos 3 eixos (sem isto a mesa e as caixas aparecem deformadas)."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    center, half = (lo + hi) / 2, (hi - lo) / 2
    for set_limits, c, h in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, half):
        set_limits(c - h, c + h)
    ax.set_box_aspect(np.maximum(hi - lo, 1e-3))


def link_views(fig, axes):
    """Rodar um dos subplots 3D com o rato roda também os outros (ficam com o mesmo ponto de vista)."""
    def on_move(event):
        if event.inaxes in axes and event.button is not None:
            for ax in axes:
                if ax is not event.inaxes:
                    ax.view_init(elev=event.inaxes.elev, azim=event.inaxes.azim)
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", on_move)


def show_figures(figures):
    """Mostra as janelas de make_figures e espera que sejam fechadas (quando uma parte corre sozinha)."""
    animations = [f["start"]() for f in figures if f.get("start")]    # guardadas, senão param
    plt.show()
    return animations


def figure_data(r):
    """Dados da janela da Parte 1 (r = resultado de run_registration): uma amostra da nuvem agregada à
    volta da mesa, a cor de cada ponto, a câmara que o adquiriu e o centro de cada câmara."""
    points = np.asarray(r["cloud"].points)
    colors = np.asarray(r["cloud"].colors)
    registered = np.array(r["registered"], bool)
    centers = np.array([T[:3, 3] for T in r["T_cams"]])
    near = np.nonzero(np.linalg.norm(points - centers[registered].mean(axis=0), axis=1) < DISPLAY_RADIUS)[0]
    idx = near[sample_indices(len(near), DISPLAY_POINTS)]
    return {"parte": "parte1_registo", "cena": Path(r["scene_folder"]).name, "points": points[idx],
            "colors": colors[idx], "camera": r["camera_index"][idx], "camera_centers": centers,
            "images": np.array(r["images"]), "registered": registered}


def make_figures(data):
    """Janela da Parte 1, como a figura do enunciado: a mesma nuvem agregada com a cor de cada píxel
    (esquerda) e com a cor da câmara que adquiriu cada ponto (direita)."""
    points = camera_to_display(data["points"])
    centers = camera_to_display(data["camera_centers"])
    registered = np.asarray(data["registered"], bool)
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"Parte 1 - registo: nuvem agregada ({data['cena']}, {len(centers)} imagens)")
    axes = [fig.add_subplot(1, 2, k, projection="3d") for k in (1, 2)]
    scatter_3d(axes[0], points, data["colors"])
    axes[0].set_title("cor de cada píxel")
    scatter_3d(axes[1], points, CAMERA_COLORS[data["camera"] % 10])
    axes[1].set_title("cor da câmara que adquiriu o ponto")
    for k, (center, name) in enumerate(zip(centers, data["images"])):
        if not registered[k]:
            continue
        for ax in axes:                      # posição de cada câmara (triângulo com o nome da imagem)
            ax.scatter(*center, color=CAMERA_COLORS[k % 10], s=60, marker="^", depthshade=False)
            ax.text(*center, f" {name}", color=CAMERA_COLORS[k % 10], fontsize=9)
    for ax in axes:
        set_equal_3d(ax, np.vstack([points, centers[registered]]))
        ax.view_init(elev=30, azim=-60)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_zlabel("-y (m)")
    left_out = [str(n) for n, ok in zip(data["images"], registered) if not ok]
    if left_out:
        fig.text(0.5, 0.03, "câmaras sem registo de confiança (fora da nuvem): " + ", ".join(left_out),
                 ha="center", color="red")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.05, top=0.92, wspace=0.02)
    link_views(fig, axes)
    return [{"name": "", "fig": fig}]


# Janelas 3D do Open3D (opção --open3d)
def show_registration(scene, T_est, camera_index_cloud=None, T_gt=None):
    """Nuvem agregada: cores reais | cor por câmara | (se houver GT) estimada a vermelho sobre a verdadeira a verde."""
    cloud, cam = aggregate(scene, T_est)
    cloud = cloud.voxel_down_sample(0.005)
    o3d.visualization.draw_geometries([cloud], window_name="Nuvem agregada (cores reais)")

    by_camera = o3d.geometry.PointCloud()
    for k, (view, T) in enumerate(zip(scene.views, T_est)):
        c = copy.deepcopy(view.cloud).voxel_down_sample(0.005)
        c.transform(T)
        c.paint_uniform_color(CAMERA_COLORS[k % 10])
        by_camera += c
    o3d.visualization.draw_geometries([by_camera], window_name="Nuvem agregada (cor = câmara)")

    if T_gt is not None:
        est, _ = aggregate(scene, T_est)
        gt, _ = aggregate(scene, T_gt)
        est = est.voxel_down_sample(0.005)
        gt = gt.voxel_down_sample(0.005)
        est.paint_uniform_color((0.9, 0.1, 0.1))
        gt.paint_uniform_color((0.1, 0.7, 0.2))
        o3d.visualization.draw_geometries([gt, est], window_name="Estimada (vermelho) vs verdadeira (verde)")


# #####################################################################################################
# 5. PROGRAMA PRINCIPAL
# #####################################################################################################
def run_registration(scene_folder, output=None, loop_closure=True, gt_registration=False, verbose=True):
    """Parte 1 numa cena: regista as imagens, junta as nuvens e grava o registo.json.

    Entrada: pasta da cena e pasta de saída (por omissão <cena>/partes).
    gt_registration=True usa as poses VERDADEIRAS das câmaras em vez do registo. Só serve para
    avaliar a segmentação e o ajuste isoladamente (como pede o enunciado); nunca no resultado final.
    Saída: dicionário com a cena, as poses das câmaras, as câmaras registadas, a nuvem agregada (e a
    câmara de cada ponto) e o ganho e o desvio de cada imagem (para o extra_fotometria.py).
    """
    t0 = time.time()
    scene = load_scene(scene_folder)
    output = Path(output) if output else Path(scene_folder) / OUTPUT_FOLDER
    if verbose:
        print(f"Cena {scene.folder.name}: {len(scene.views)} imagens")

    if gt_registration:
        T_cams = [np.array(c["T_camera0_camera"]) for c in scene.ground_truth["cameras"]]
        T_chain = T_cams
        registered = [True] * len(T_cams)
    else:
        T_cams, T_chain, registered = register_scene(scene, loop_closure=loop_closure, verbose=verbose)
    if verbose and not all(registered):
        left_out = [v.name for v, ok in zip(scene.views, registered) if not ok]
        print(f"  câmaras sem registo de confiança, deixadas de fora da nuvem: {', '.join(left_out)}")

    exposures = []                       # (imagem, s, b) de cada câmara, preenchido pelo aggregate
    cloud, camera_index = aggregate(scene, T_cams, registered, exposures=exposures)
    save_registration(output, scene, T_cams, T_chain, registered)
    if verbose:
        print(f"  nuvem agregada com {len(cloud.points)} pontos; registo gravado em "
              f"{output / REGISTRATION_FILE} ({time.time() - t0:.1f} s)")
    return {"scene": scene, "scene_folder": scene.folder, "images": [v.name for v in scene.views],
            "T_cams": T_cams, "T_chain": T_chain, "registered": registered, "cloud": cloud,
            "camera_index": camera_index, "exposures": exposures, "folder": output}


def main():
    """Lê os argumentos da linha de comandos e corre a Parte 1."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="pasta da cena")
    parser.add_argument("--no_loop_closure", action="store_true", help="registo só por encadeamento")
    parser.add_argument("--gt_registration", action="store_true",
                        help="usa as poses verdadeiras das câmaras (só para avaliar o resto do sistema)")
    parser.add_argument("--visualize", action="store_true", help="janela com a nuvem agregada (2 subplots)")
    parser.add_argument("--open3d", action="store_true",
                        help="janelas 3D do Open3D (com ground truth: nuvem estimada sobre a verdadeira)")
    args = parser.parse_args()

    r = run_registration(args.scene, loop_closure=not args.no_loop_closure, gt_registration=args.gt_registration)
    print(f"\nRegisto gravado em {r['folder']}. A seguir:\n"
          f"    python savi/extra_fotometria.py --scene {args.scene}\n"
          f"    python savi/parte2_segmentacao.py --scene {args.scene} --num_boxes N")
    if args.visualize:
        show_figures(make_figures(figure_data(r)))
    if args.open3d:
        gt = r["scene"].ground_truth
        T_gt = [np.array(c["T_camera0_camera"]) for c in gt["cameras"]] if gt else None
        show_registration(r["scene"], r["T_cams"], T_gt=T_gt)


if __name__ == "__main__":
    main()
