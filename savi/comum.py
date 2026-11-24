#!/usr/bin/env python3
"""SAVIBoxes - código COMUM a todas as partes (não pertence a nenhuma parte em particular).

    1. Geometria        matrizes 4x4, rotações e caixas
    2. Leitura da cena  ler as imagens e as nuvens de pontos, ler o ground truth e escrever o result.json

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

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2                                   # OpenCV: ler as imagens
import numpy as np
import open3d as o3d                         # nuvens de pontos


OUTPUT_FOLDER = "partes"      # pasta, dentro da cena, onde cada parte grava o seu resultado


# #####################################################################################################
# 1. GEOMETRIA
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
# 2. LEITURA DA CENA E ESCRITA DO RESULTADO
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
