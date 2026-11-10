"""Leitura das cenas no formato do simulador e escrita do ficheiro result.json.

Formato de uma cena (igual para cenas simuladas e reais):
    scene/
        point_clouds/000.ply   nuvem de pontos com cor, no referencial da câmara que a adquiriu
        rgb/000.png            imagem a cores correspondente
        intrinsics.json        fx, fy, cx, cy, width, height
        ground_truth.json      (só para avaliação)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


@dataclass
class View:
    """Uma imagem RGB-D da cena: nome, imagem a cores, nuvem de pontos e 'imagem de pontos 3D'."""
    name: str                      # nome da imagem, por exemplo "000"
    rgb: np.ndarray                # imagem a cores (H x W x 3, RGB, uint8)
    cloud: o3d.geometry.PointCloud  # nuvem de pontos com cor, no referencial da câmara
    xyz_image: np.ndarray          # H x W x 3: ponto 3D de cada pixel (0 onde não há medição)


@dataclass
class Scene:
    """Uma cena completa: as imagens, os parâmetros intrínsecos e, se existir, o ground truth."""
    folder: Path
    views: list
    K: np.ndarray                  # matriz de parâmetros intrínsecos 3x3
    width: int
    height: int
    ground_truth: dict = field(default=None)


def intrinsics_matrix(intrinsics):
    """Constrói a matriz K a partir de fx, fy, cx, cy."""
    return np.array([[intrinsics["fx"], 0.0, intrinsics["cx"]],
                     [0.0, intrinsics["fy"], intrinsics["cy"]],
                     [0.0, 0.0, 1.0]])


def cloud_to_xyz_image(cloud, K, width, height):
    """Reconstrói a 'imagem de pontos 3D' a partir da nuvem de pontos.

    Cada ponto da nuvem foi criado a partir de um pixel (u, v) da imagem de profundidade.
    Projetando o ponto com a matriz K recuperamos esse pixel: u = fx*x/z + cx, v = fy*y/z + cy.
    Assim conseguimos saber, para um pixel da imagem a cores (por exemplo uma feature SIFT),
    qual é o ponto 3D correspondente. É isto que permite passar de associações 2D para pares 3D.
    """
    points = np.asarray(cloud.points)
    xyz_image = np.zeros((height, width, 3), np.float64)
    z = points[:, 2]
    valid = z > 0
    u = np.round(K[0, 0] * points[valid, 0] / z[valid] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * points[valid, 1] / z[valid] + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    xyz_image[v[inside], u[inside]] = points[valid][inside]
    return xyz_image


def load_scene(folder):
    """Lê todas as imagens e nuvens de pontos de uma pasta de cena."""
    folder = Path(folder)
    intrinsics = json.loads((folder / "intrinsics.json").read_text(encoding="utf-8"))
    K = intrinsics_matrix(intrinsics)
    width, height = int(intrinsics["width"]), int(intrinsics["height"])

    views = []
    # As nuvens de pontos estão ordenadas pelo nome: 000.ply, 001.ply, ...
    for ply in sorted((folder / "point_clouds").glob("*.ply")):
        name = ply.stem
        cloud = o3d.io.read_point_cloud(str(ply))
        # cv2.imread falha no Windows com caminhos com acentos (ex.: "Trabalho prático"),
        # por isso lemos os bytes com numpy e descodificamos com cv2.imdecode.
        # O OpenCV devolve BGR; convertemos para RGB para ficar igual às cores da nuvem.
        data = np.fromfile(str(folder / "rgb" / f"{name}.png"), dtype=np.uint8)
        rgb = cv2.imdecode(data, cv2.IMREAD_COLOR)[..., ::-1].copy()
        views.append(View(name, rgb, cloud, cloud_to_xyz_image(cloud, K, width, height)))

    # O ground truth só é lido se existir; nunca é usado para estimar nada, só para avaliar
    ground_truth = None
    gt_path = folder / "ground_truth.json"
    if gt_path.exists():
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))

    return Scene(folder, views, K, width, height, ground_truth)


def write_result(path, view_names, T_camera0_cameras, boxes):
    """Escreve o result.json no formato pedido no enunciado.

    boxes: lista de dicionários com "T_camera0_box" (4x4), "length", "width" e "height".
    """
    result = {
        "cameras": [{"image": name, "T_camera0_camera": np.round(T, 6).tolist()}
                    for name, T in zip(view_names, T_camera0_cameras)],
        "boxes": [{"T_camera0_box": np.round(b["T_camera0_box"], 6).tolist(),
                   "length": round(float(b["length"]), 6),
                   "width": round(float(b["width"]), 6),
                   "height": round(float(b["height"]), 6)} for b in boxes],
    }
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
