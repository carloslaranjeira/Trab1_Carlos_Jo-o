#!/usr/bin/env python3
"""SAVIBoxes - Trabalho 1 de SAVI. PARTE 1: REGISTO (pose de cada câmara e nuvem de pontos agregada).

=====================================================================================================
COMO CORRER
=====================================================================================================
    (a partir da pasta principal do trabalho)
    python savi/parte1_registo.py --scene test/cena_1
    python savi/parte1_registo.py --scene test/cena_1 --visualize   # janelas 3D: nuvem agregada
    python savi/parte1_registo.py --scene test/cena_1 --open3d      # janelas antigas do Open3D

Não precisa do número de caixas. Grava <cena>/partes/registo.json com a pose de cada câmara em relação
à câmara 000 (T_camera0_camera) e as câmaras que entraram na nuvem agregada.

Ordem das partes (o run_test.py corre-as todas, cena a cena):
    parte1_registo.py      -> <cena>/partes/registo.json   (pose de cada câmara)
    extra_fotometria.py    -> <cena>/partes/fotometria.json e fotometria/ (s e b, imagens corrigidas)
    parte2_segmentacao.py  -> <cena>/partes/objetos.ply, mesa.json, segmentacao.npz (estima o n.º de caixas)
    parte3_ajuste.py       -> result.json e <cena>/partes/ajuste.npz
    avaliador.py           -> pede o n.º real de caixas e compara com o ground truth (se a cena o tiver)

=====================================================================================================
ORGANIZAÇÃO DESTE FICHEIRO (as 4 subpartes do enunciado, pela mesma ordem)
=====================================================================================================
    1. Pré-processamento   downsampling (voxel), remoção de outliers e normais        (Parte 07, Ex. 4 e 5)
    2. Estimativa inicial  features SIFT das imagens + teste de Lowe (Parte 04) -> pares de pontos 3D ->
                           RANSAC do Open3D, robusto a associações erradas            (Parte 08, Ex. 3)
    3. ICP                 registration_icp ponto-a-plano com a nuvem COMPLETA, do limiar grosseiro para
                           o fino, e encadeamento das transformações                  (Parte 08, Ex. 2)
    4. Agregação           as transformações aplicadas às nuvens com a resolução original, e as figuras
    5. Extras de robustez  NÃO fazem parte das 4 subpartes: escolha entre várias hipóteses, hipóteses
                           a partir da mesa, fecho de ciclo e grafo de poses
    6. Programa principal  junta as subpartes por ordem: register_pair (um par de câmaras),
                           register_scene (todas), run_registration (+ registo.json) e main()

Código que não é da Parte 1 (usado por todas as partes):
    comum.py          geometria (matrizes 4x4, caixas), leitura da cena, result.json, convenções
    visualizacao.py   janelas 3D do Open3D (SceneWindows) e elementos a desenhar
"""

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2                                   # OpenCV: features das imagens
import numpy as np
import open3d as o3d                         # nuvens de pontos: RANSAC, ICP, grafo de poses
from scipy.spatial import cKDTree            # procura rápida do vizinho mais próximo

from comum import intrinsics_matrix, OUTPUT_FOLDER, load_scene, rotation_angle_deg, rz4, transform_points
# EXTRA fotometria: comparar cores de imagens com exposições diferentes (alignment_score)
from extra_fotometria import exposure_corrected
from visualizacao import (  # janelas 3D (comuns a todas as partes)
    CAMERA_COLORS, CAMERA_NAMES, camera_item, colors_uint8, data_view, points_item, scene_view, show_windows,
)


reg = o3d.pipelines.registration             # atalho para o módulo de registo do Open3D

# Parâmetros (valores escolhidos com testes nas cenas do simulador)
ICP_VOXELS = (0.04, 0.02, 0.01, 0.005)   # 1/3. voxels do pré-processamento = escalas do ICP, do grosseiro para o fino (m)
MAX_DEPTH = 3.0                          # 1. filtro: pontos a mais de 3 m da câmara (ruído ~2 cm) não entram no ICP
LOWE_RATIO = 0.8                         # 2. teste da razão de Lowe (Parte 04, Ex. 4)
RANSAC_THRESHOLD = 0.02                  # 2. distância máxima (m) para um par de pontos 3D ser inlier
RANSAC_ITERATIONS = 1_000_000            # 2. iterações do RANSAC (com ~4% de inliers são precisas muitas)


@dataclass
class Camera:
    """Tudo o que se calcula UMA vez por imagem, antes de registar os pares."""
    view: object                     # a View da cena (imagem, nuvem e imagem de pontos 3D), ver comum.py
    clouds: dict                     # 1. nuvem pré-processada para cada voxel do ICP: {voxel: nuvem}
    small: object                    # 1. nuvem completa reduzida a 2 cm (para avaliar alinhamentos, extra)
    features: tuple                  # 2. (keypoints, descritores) SIFT
    table: object = None             # 5. mesa vista só por esta câmara (extra), ou None
    asift: tuple = field(default=None)   # 2. features ASIFT, só calculadas se o SIFT não chegar

    @property
    def name(self):
        return self.view.name


# #####################################################################################################
# 1. PRÉ-PROCESSAMENTO
# #####################################################################################################
# Filtrar os pontos, fazer downsampling e estimar as normais (Parte 07, Ex. 4 e 5). Faz-se uma vez por
# imagem e por escala do ICP; o ICP (subparte 3) usa sempre estas nuvens.
def preprocess(cloud, voxel):
    """Prepara uma nuvem para o ICP: downsampling, remoção de outliers e normais.

    - voxel_down_sample: substitui os pontos de cada cubo (voxel) pela sua média -> densidade
      uniforme e ICP muito mais rápido (Parte 07, Ex. 4);
    - remove_statistical_outlier: retira pontos com distância média aos vizinhos anormal (os 'flying
      pixels' das arestas, que o simulador acrescenta). Não está nas aulas: tutorial "Point cloud
      outlier removal" do Open3D;
    - estimate_normals: o ICP ponto-a-plano precisa da normal de cada ponto (Parte 07, Ex. 5).
    """
    down = cloud.voxel_down_sample(voxel)
    down, _ = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return down


def preprocess_view(view, voxels=ICP_VOXELS, max_depth=None):
    """Pré-processamento de uma imagem: a nuvem preparada para cada escala do ICP.

    Filtro por profundidade: ficam só os pontos a menos de max_depth metros da câmara (o z da câmara é a
    profundidade). O ruído da profundidade cresce com z^2 (a 4 m são ~2.5 cm) e esses pontos estragam o
    ICP: sem este filtro o erro do registo foi até 24 vezes maior (level_2 com 5 câmaras: 1.3 mm com o
    filtro, 30 mm sem ele). A mesa, as caixas e o fundo próximo (paredes e móveis a menos de 3 m)
    continuam todos na nuvem, como o enunciado pede. max_depth=None não corta nada.
    Saída: {voxel: nuvem pré-processada}.
    """
    cloud = view.cloud
    if max_depth is not None:
        z = np.asarray(cloud.points)[:, 2]
        cloud = cloud.select_by_index(np.nonzero(z < max_depth)[0])
    return {voxel: preprocess(cloud, voxel) for voxel in voxels}


# #####################################################################################################
# 2. ESTIMATIVA INICIAL
# #####################################################################################################
# Features das imagens (Parte 04) -> pares de pontos 3D (nota da Parte 08, Ex. 3) -> transformação
# inicial por RANSAC, robusta a associações erradas.
def detect_features(view, method="sift"):
    """Deteta keypoints e descritores na imagem a cores (só onde há medição 3D).

    method="sift":  cv2.SIFT_create, como na Parte 04 (Ex. 2). Sem limite de keypoints (nfeatures=0),
                    para haver mais associações entre câmaras afastadas;
    method="asift": Affine-SIFT (cv2.AffineFeature, não está nas aulas). Simula várias inclinações da
                    imagem antes de calcular os descritores e encontra 3 a 10 vezes mais associações
                    corretas entre câmaras a 45°. É mais lento, por isso só se usa quando o SIFT não chega.
    A máscara evita features em pixels sem profundidade (não dariam pontos 3D).
    Saída: (keypoints, descritores), como o detectAndCompute do OpenCV.
    """
    detector = cv2.SIFT_create(nfeatures=0)
    if method == "asift":
        detector = cv2.AffineFeature_create(detector)
    gray = cv2.cvtColor(view.rgb, cv2.COLOR_RGB2GRAY)
    mask = (view.xyz_image[..., 2] > 0).astype(np.uint8) * 255
    return detector.detectAndCompute(gray, mask)


def match_3d_pairs(view_a, feats_a, view_b, feats_b, ratio=LOWE_RATIO):
    """Associa features entre duas imagens e devolve os PARES DE PONTOS 3D correspondentes.

    1. Para cada descritor de B procura os 2 mais parecidos em A com o FLANN (knnMatch, k=2), como na
       Parte 04 (Ex. 3 e 4), e aplica o teste da razão de Lowe: o melhor tem de ser claramente melhor
       do que o segundo, senão a associação é ambígua e é descartada;
    2. Para cada associação 2D vai buscar o ponto 3D desse pixel em cada imagem (xyz_image).
    Saída: (pontos em A, pontos em B), ambos N x 3, com pontos_A[i] <-> pontos_B[i].
    """
    (kp_a, des_a), (kp_b, des_b) = feats_a, feats_b
    if des_a is None or des_b is None or len(kp_a) < 3 or len(kp_b) < 3:
        return np.zeros((0, 3)), np.zeros((0, 3))
    height, width = view_a.xyz_image.shape[:2]

    # FLANN com árvores kd (algorithm=1), como na Parte 04
    flann_matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    two_best_matches = flann_matcher.knnMatch(des_b, des_a, k=2)
    points_a, points_b = [], []
    for pair in two_best_matches:
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


def is_different(T1, T2, max_translation=0.05, max_angle=5.0):
    """True se duas transformações são soluções diferentes (diferem mais de 5 cm ou mais de 5°)."""
    return (np.linalg.norm(T1[:3, 3] - T2[:3, 3]) > max_translation or
            rotation_angle_deg(T1[:3, :3].T @ T2[:3, :3]) > max_angle)


def ransac_hypotheses(points_src, points_dst, threshold=RANSAC_THRESHOLD, iterations=RANSAC_ITERATIONS,
                      max_hypotheses=4, seed=0):
    """RANSAC: as melhores hipóteses DIFERENTES de T_dst_src a partir de pares de pontos 3D.

    Usa o RANSAC do Open3D com correspondências conhecidas (registration_ransac_based_on_correspondence,
    do tutorial de registo global para o qual a Parte 08, Ex. 3 remete). Em cada iteração escolhe 3
    pares, calcula a transformação rígida (TransformationEstimationPointToPoint, método de Kabsch) e
    conta os pares que ficam a menos de 'threshold' metros (inliers). É robusto a associações erradas,
    como o findHomography com RANSAC da Parte 04 (Ex. 5) e da Parte 06 (Ex. 5).
    Os 'checkers' rejeitam amostras antes de contar os inliers:
      - EdgeLength: uma transformação rígida preserva distâncias, por isso os 3 pares têm de formar
        triângulos com lados parecidos nas duas nuvens;
      - Distance: os 3 pares têm de ficar a menos de 'threshold' depois de transformados.

    Porquê várias hipóteses? Texturas repetidas (tábuas do chão, livros) geram grupos de associações
    erradas mas coerentes entre si; às vezes um modelo errado tem mais inliers do que o certo.
    Como na deteção sucessiva de planos da Parte 07 (Ex. 3), depois de cada hipótese retiram-se os
    seus inliers e corre-se o RANSAC outra vez com os pares que sobram. Quem decide entre as
    hipóteses é o ICP + o score (extras, secção 5).
    Saída: lista de (n.º de inliers, T 4x4), da primeira encontrada (mais inliers) para a última.
    """
    o3d.utility.random.seed(seed)                   # resultados repetíveis
    hypotheses = []
    remaining = np.arange(len(points_src))          # pares ainda não explicados por nenhuma hipótese
    while len(hypotheses) < max_hypotheses and len(remaining) >= 3:
        source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_src[remaining]))
        target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_dst[remaining]))
        pairs = np.arange(len(remaining))
        correspondences = o3d.utility.Vector2iVector(np.stack([pairs, pairs], axis=1))   # i <-> i
        result = reg.registration_ransac_based_on_correspondence(
            source, target, correspondences, threshold,
            reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(threshold)],
            reg.RANSACConvergenceCriteria(iterations, 0.9999))
        inliers = np.asarray(result.correspondence_set).reshape(-1, 2)[:, 0]
        if len(inliers) < 3:
            break
        T = np.array(result.transformation)
        if all(is_different(T, U) for _, U in hypotheses):
            hypotheses.append((len(inliers), T))
        remaining = np.delete(remaining, inliers)   # o RANSAC seguinte só usa os pares que sobram
    return hypotheses


# #####################################################################################################
# 3. ICP
# #####################################################################################################
# Refinar as transformações com o registration_icp do Open3D (Parte 08, Ex. 2), sempre com a nuvem
# completa (mesa e fundo), e encadear as transformações dos pares de câmaras vizinhas.
def multiscale_icp(cam_src, cam_dst, T_init, voxels=ICP_VOXELS):
    """ICP ponto-a-plano em várias escalas, da grosseira para a fina (Parte 08, Ex. 2).

    Entrada: câmara fonte, câmara alvo, estimativa inicial T_alvo_fonte, lista de voxels.
    Em cada escala usa as nuvens do pré-processamento com esse voxel, e a distância máxima de
    correspondência é 1.5 x o voxel. É o que a Parte 08 (Ex. 2.3) pede: correr de novo o ICP a partir
    do resultado, com um limiar menor. Começar grosseiro dá uma bacia de convergência larga; acabar
    fino dá precisão.
    O kernel de Tukey (não está nas aulas: tutorial "Robust kernels" do Open3D) dá peso zero aos
    resíduos grandes: pontos muito ruidosos ou zonas que só uma das câmaras vê não puxam a solução.
    Saída: (T final, resultado do Open3D com fitness/rmse, matriz de informação para o grafo de poses).
    """
    T = T_init
    for voxel in voxels:
        src, dst = cam_src.clouds[voxel], cam_dst.clouds[voxel]
        estimation = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=voxel))
        result = reg.registration_icp(src, dst, voxel * 1.5, T, estimation,
                                      reg.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7,
                                                                 max_iteration=60))
        T = result.transformation
    information = reg.get_information_matrix_from_point_clouds(src, dst, voxels[-1] * 1.5, T)
    return T, result, information


def chain_poses(edges):
    """Encadeamento: T_0_(i+1) = T_0_i @ T_i_(i+1), a partir das transformações dos pares (i, i+1).

    O erro de cada par acumula-se ao longo da cadeia (deriva): é uma das perguntas do enunciado.
    Entrada: arestas (i, i+1, T_i_(i+1), ...) pela ordem das câmaras. Saída: T_camera0_camera de cada câmara.
    """
    poses = [np.eye(4)]
    for i, j, T, *_ in edges:
        poses.append(poses[i] @ T)
    return poses


# #####################################################################################################
# 4. AGREGAÇÃO
# #####################################################################################################
def aggregate(scene, T_camera0_cameras, registered=None):
    """Junta as nuvens com a RESOLUÇÃO ORIGINAL no referencial da câmara 0.

    Cada nuvem é copiada, transformada com T_camera0_camera (leva os pontos para o referencial da
    câmara 0) e somada à nuvem agregada.
    registered: lista de True/False por câmara; as câmaras com False (registo falhado) ficam de fora.
    As cores são as das nuvens da cena: depois do extra_fotometria.py, as das imagens corrigidas.
    Saída: (nuvem agregada, índice da câmara que adquiriu cada ponto).
    """
    aggregated = o3d.geometry.PointCloud()
    camera_index = []
    for k, (view, T) in enumerate(zip(scene.views, T_camera0_cameras)):
        if registered is not None and not registered[k]:
            continue
        cloud = copy.deepcopy(view.cloud)       # cópia, para não alterar a nuvem original
        cloud.transform(T)                      # leva os pontos para o referencial da câmara 0
        aggregated += cloud
        camera_index.append(np.full(len(cloud.points), k))
    return aggregated, np.concatenate(camera_index)


# ----------------------------------------------------------------------------------------------------
# Figuras da nuvem agregada (as janelas são as de visualizacao.py)
# ----------------------------------------------------------------------------------------------------
def figure_data(r):
    """Dados das janelas da Parte 1 (r = resultado de run_registration): a nuvem agregada inteira, a cor
    de cada ponto, a câmara que o adquiriu, a pose e os parâmetros intrínsecos das câmaras (para as
    desenhar) e o ponto de vista (atrás da câmara 000)."""
    view = scene_view(r["T_cams"], r["registered"])
    if r.get("scene") is not None:
        K, size = r["scene"].K, (r["scene"].width, r["scene"].height)
    else:                                        # resultado de load_registration: lê só o intrinsics.json
        intrinsics = json.loads((Path(r["scene_folder"]) / "intrinsics.json").read_text(encoding="utf-8"))
        K, size = intrinsics_matrix(intrinsics), (intrinsics["width"], intrinsics["height"])
    return {"parte": "parte1_registo", "cena": Path(r["scene_folder"]).name,
            "points": np.asarray(r["cloud"].points, np.float32), "colors": colors_uint8(r["cloud"].colors),
            "camera": np.asarray(r["camera_index"], np.uint8), "images": np.array(r["images"]),
            "registered": np.array(r["registered"], bool), "T_cams": np.array(r["T_cams"]),
            "K": np.asarray(K, float), "image_size": np.array(size), **{f"view_{k}": v for k, v in view.items()}}


def make_windows(data):
    """Janelas da Parte 1, como a figura do enunciado: a nuvem agregada com a cor de cada píxel e a mesma
    nuvem com uma cor por câmara (cada uma na sua janela). Na segunda, cada câmara registada aparece
    também na sua posição (uma pirâmide virada para onde a câmara olha), com a cor dos seus pontos."""
    left_out = [str(n) for n, ok in zip(data["images"], data["registered"]) if not ok]
    note = f" (sem registo de confiança, fora da nuvem: {', '.join(left_out)})" if left_out else ""
    title = f"Parte 1 - {data['cena']}: nuvem agregada"
    cameras = ", ".join(f"{n}: {CAMERA_NAMES[k % 10]}" for k, n in enumerate(data["images"]) if data["registered"][k])
    width, height = (int(v) for v in data["image_size"])
    camera_items = [camera_item(T, data["K"], width, height, CAMERA_COLORS[k % 10])
                    for k, T in enumerate(data["T_cams"]) if data["registered"][k]]
    # Ponto de vista que mostra as câmaras todas: a olhar para o mesmo sítio, mas mais para trás e para
    # cima, do lado das câmaras (2 vezes a distância média do centro às câmaras, e outra vez essa distância
    # acima). Com câmaras à volta da mesa toda, a média das câmaras fica no centro: usa-se o lado da 000.
    view = data_view(data)
    center = np.asarray(view["center"], float)
    up = np.asarray(view["up"], float) / np.linalg.norm(view["up"])
    positions = np.asarray(data["T_cams"])[data["registered"]][:, :3, 3] - center
    radius = float(np.linalg.norm(positions, axis=1).mean())
    direction = positions.mean(axis=0)
    if np.linalg.norm(direction) < 0.3 * radius:
        direction = positions[0]
    direction = direction / np.linalg.norm(direction)
    cameras_view = {"center": center, "up": up, "eye": center + radius * (2.0 * direction + up)}
    return [{"name": "_cor_pixeis", "title": f"{title} com a cor dos píxeis{note}",
             "items": [points_item(data["points"], data["colors"])], "view": data_view(data)},
            {"name": "_cor_camaras", "title": f"{title} com uma cor por câmara e a posição de cada câmara "
                                              f"({cameras}){note}",
             "items": [points_item(data["points"], CAMERA_COLORS[data["camera"] % 10])] + camera_items,
             "view": cameras_view}]


def show_registration(scene, T_est):
    """Janelas antigas do Open3D (opção --open3d): nuvem agregada com as cores reais | uma cor por câmara.
    (A nuvem estimada sobre a verdadeira é desenhada pelo avaliador.py.)"""
    cloud, _ = aggregate(scene, T_est)
    cloud = cloud.voxel_down_sample(0.005)
    o3d.visualization.draw_geometries([cloud], window_name="Nuvem agregada (cores reais)")

    by_camera = o3d.geometry.PointCloud()
    for k, (view, T) in enumerate(zip(scene.views, T_est)):
        c = copy.deepcopy(view.cloud).voxel_down_sample(0.005)
        c.transform(T)
        c.paint_uniform_color(CAMERA_COLORS[k % 10])
        by_camera += c
    o3d.visualization.draw_geometries([by_camera], window_name="Nuvem agregada (cor = câmara)")


# #####################################################################################################
# 5. EXTRAS DE ROBUSTEZ (não fazem parte das 4 subpartes do enunciado)
# #####################################################################################################
# Com poucas câmaras (45° a 75° entre vizinhas) as features dão poucas associações corretas e o RANSAC
# pode escolher uma transformação errada. Estes extras geram mais hipóteses, escolhem a melhor e
# rejeitam os pares sem confiança (vêm de testes com as cenas do simulador, por exemplo os testes 10 e 16):
#   5.1 avaliar e escolher entre as hipóteses de um par (score com cor, espaço livre, confiança);
#   5.2 hipóteses a partir da mesa vista por cada câmara sozinha;
#   5.3 fecho de ciclo, câmaras isoladas e grafo de poses (tutorial "Multiway registration" do Open3D).
# Notas sobre as escolhas:
#   - Features 3D (FPFH, Parte 08 Ex. 3) falham nesta cena: está cheia de planos (mesa, chão, paredes),
#     que têm todos o mesmo descritor, e o RANSAC alinha os planos errados.
MIN_PAIR_SCORE = 0.18   # score >= 0.18: o par é sempre de confiança (pares corretos: 0.26-0.48)
LOW_PAIR_SCORE = 0.05   # entre 0.05 e 0.18: só se for claramente melhor que as alternativas
MARGIN_RATIO = 1.5      # 'claramente melhor' = score >= 1.5 x o score da melhor hipótese DIFERENTE
MAX_VIOLATION = 0.10    # fração máxima de pontos em 'espaço livre' (corretas 0.01-0.055, erradas 0.19-0.40)
CONFIRMED_RATIO = 0.75  # hipótese confirmada pela mesa E pelas features: aceita se score >= 0.75 x o melhor
MAX_LOOP_ANGLE = 100.0  # graus: câmaras com direções de vista mais afastadas não se sobrepõem
MIN_CAMERA_HEIGHT = 0.1 # m: cada câmara tem de ficar pelo menos 10 cm acima do tampo visto pela outra


# ----------------------------------------------------------------------------------------------------
# 5.1 Avaliar e escolher entre as hipóteses de um par
# ----------------------------------------------------------------------------------------------------
def alignment_score(cam_src, cam_dst, T, distance=0.02, max_color_diff=0.08):
    """Nota de 0 a 1 de um alinhamento, usando as nuvens COMPLETAS (com o fundo) e a cor.

    Saída: fração de pontos da fonte que, depois de transformados por T, têm um ponto do alvo a
    menos de 'distance' metros E com cor parecida.
    Porquê não usar só o fitness do ICP? A Parte 08 pergunta se o fitness chega para saber se o
    alinhamento está certo: não chega. Um alinhamento errado que faz 'deslizar' a mesa sobre si
    própria tem boa sobreposição local, mas falha no fundo e na cor (põe chão em cima da mesa).
    """
    src = copy.deepcopy(cam_src.small)        # cópia: o transform altera a nuvem
    dst = cam_dst.small
    src.transform(T)
    tree = cKDTree(np.asarray(dst.points))    # para achar o vizinho mais próximo
    dist, idx = tree.query(np.asarray(src.points), k=1)
    src_colors = exposure_corrected(np.asarray(src.colors), np.asarray(dst.colors)[idx], dist < distance)
    color_diff = np.abs(np.asarray(dst.colors)[idx] - src_colors).mean(axis=1)
    good = (dist < distance) & (color_diff < max_color_diff)
    return good.mean()


def free_space_violation(view_target, K, T_target_source, cam_source, margin=0.03):
    """Fração de pontos da fonte que ficam em ESPAÇO LIVRE visto pela câmara alvo.

    Ideia: cada pixel da câmara alvo mede a distância até à primeira superfície no seu raio; o
    espaço entre a câmara e essa superfície está VAZIO. Depois de transformados, os pontos da fonte
    não podem ficar nesse espaço vazio (a câmara alvo tê-los-ia visto à frente).
    Como: projeta-se cada ponto na imagem do alvo (com K) e contam-se os que ficam mais de
    'margin' metros À FRENTE da profundidade medida nesse pixel.
    Um alinhamento correto tem poucos (só ruído nas arestas); um errado põe, por exemplo, a mesa a
    atravessar o chão. Funciona mesmo quando as duas vistas quase não se sobrepõem.
    """
    points = transform_points(T_target_source, np.asarray(cam_source.small.points))
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


def camera_above_table(cam_a, T_a_b, min_height=MIN_CAMERA_HEIGHT):
    """True se, com a hipótese T_a_b, a câmara b fica ACIMA do tampo da mesa visto pela câmara a.

    As câmaras olham para as caixas que estão sobre a mesa, por isso estão sempre acima do tampo (nas
    cenas do simulador, 0.65 a 0.76 m). As hipóteses erradas vindas de texturas repetidas do chão ou das
    paredes punham a outra câmara 2.2 m ABAIXO da mesa, debaixo do chão, e passavam nos outros testes
    (level_4 com 2 câmaras e level_3_aligned com 3). A altura é a coordenada z do centro da câmara b no
    referencial da mesa da câmara a: inv(T_a_table) @ T_a_b. Sem mesa detetada na câmara a, não se testa.
    """
    if cam_a.table is None:
        return True
    return (np.linalg.inv(cam_a.table.T_camera0_table) @ T_a_b)[2, 3] > min_height


def is_plausible(cam_a, cam_b, K, T):
    """Uma hipótese T_a_b é plausível se cada câmara ficar acima da mesa vista pela outra e se nenhuma das
    câmaras vir pontos da outra em espaço livre."""
    if not (camera_above_table(cam_a, T) and camera_above_table(cam_b, np.linalg.inv(T))):
        return False
    if K is None:
        return True
    return max(free_space_violation(cam_a.view, K, T, cam_b),
               free_space_violation(cam_b.view, K, np.linalg.inv(T), cam_a)) < MAX_VIOLATION


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
# 5.2 Hipóteses a partir da mesa vista por cada câmara (a deteção da mesa é a da Parte 2)
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
# 5.3 Fecho de ciclo, câmaras isoladas e grafo de poses
# ----------------------------------------------------------------------------------------------------
def poses_from_edges(n, edges):
    """Pose T_0_k de cada câmara, percorrendo o grafo de arestas a partir da câmara 0.

    Cada aresta (i, j, T_i_j, ...) pode ser percorrida nos dois sentidos:
        T_0_j = T_0_i @ T_i_j        ou        T_0_i = T_0_j @ inv(T_i_j)
    (pesquisa em largura). Câmaras que não se conseguem alcançar ficam com None.
    Ao contrário do encadeamento simples (chain_poses), só usa as arestas dadas: as dos pares sem
    confiança ficam de fora (um par errado estragaria todas as câmaras seguintes).
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


def extra_edges(n, pair, good, views, verbose=True):
    """Acrescenta às arestas de confiança (good) o fecho de ciclo e as arestas das câmaras isoladas.

    1. Fecho de ciclo: regista também a última imagem com a primeira, mas só se as duas olharem para
       lados parecidos (< MAX_LOOP_ANGLE). Com poucas câmaras num arco a primeira e a última olham em
       direções opostas e não têm nada em comum;
    2. Câmaras isoladas (sem pose porque um par falhou): numa volta completa cada câmara continua
       alcançável pelo outro lado; se não, tenta-se registá-la com uma câmara a 2 posições de distância.
    pair(a, b) regista a imagem b com a imagem a. Saída: (arestas, pose de cada câmara ou None).
    """
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
    return good, poses


def optimize_pose_graph(poses, edges):
    """Grafo de poses (Open3D): cada nó é uma câmara (pose T_0_i), cada aresta é o registo de um par.

    A otimização distribui o erro pelas arestas de forma ótima (com o fecho de ciclo, a deriva do
    encadeamento deixa de se acumular toda na última câmara).
    Saída: T_camera0_camera de cada câmara, com a câmara 0 exatamente na identidade.
    """
    graph = reg.PoseGraph()
    for T_node in poses:
        graph.nodes.append(reg.PoseGraphNode(T_node))
    for i, j, T, info, _, uncertain in edges:
        # O Open3D quer a aresta (source=j, target=i) com T a levar pontos de 'source' para 'target'
        graph.edges.append(reg.PoseGraphEdge(j, i, T, info, uncertain=uncertain))
    option = reg.GlobalOptimizationOption(max_correspondence_distance=ICP_VOXELS[-1] * 1.5,
                                          edge_prune_threshold=0.25, reference_node=0)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    reg.global_optimization(graph, reg.GlobalOptimizationLevenbergMarquardt(),
                            reg.GlobalOptimizationConvergenceCriteria(), option)
    T_graph = [np.array(node.pose) for node in graph.nodes]
    T0_inv = np.linalg.inv(T_graph[0])               # a câmara 0 tem de ser exatamente a identidade
    return [T0_inv @ T for T in T_graph]


# #####################################################################################################
# 6. PROGRAMA PRINCIPAL: junta as subpartes por ordem
# #####################################################################################################
REGISTRATION_FILE = "registo.json"    # pose de cada câmara (é a entrada das outras partes)


def prepare_cameras(views, verbose=True):
    """O que se calcula uma vez por imagem: pré-processamento (1), features SIFT (2) e a mesa (extra)."""
    if verbose:
        print("  1. pré-processamento das nuvens, features SIFT e mesa de cada imagem...")
    tables = view_tables(views)
    return [Camera(view=v, clouds=preprocess_view(v, max_depth=MAX_DEPTH), small=v.cloud.voxel_down_sample(0.02),
                   features=detect_features(v, "sift"), table=t) for v, t in zip(views, tables)]


def feature_candidates(cam_a, cam_b, method, ratio, max_hypotheses):
    """Subpartes 2 e 3 para um par: hipóteses das features (RANSAC), cada uma refinada com o ICP
    grosseiro e avaliada (alignment_score, extra).

    Saída: (lista de (score, T, origem), n.º de pares 3D encontrados).
    """
    if method == "asift":
        for cam in (cam_a, cam_b):
            if cam.asift is None:
                cam.asift = detect_features(cam.view, "asift")
        feats_a, feats_b = cam_a.asift, cam_b.asift
    else:
        feats_a, feats_b = cam_a.features, cam_b.features
    pts_a, pts_b = match_3d_pairs(cam_a.view, feats_a, cam_b.view, feats_b, ratio=ratio)
    hypotheses = ransac_hypotheses(pts_b, pts_a, max_hypotheses=max_hypotheses)
    candidates = []
    for inliers, T_init in hypotheses:
        # Só as escalas grosseiras do ICP: chega para escolher a hipótese e é mais rápido
        T, _, _ = multiscale_icp(cam_b, cam_a, T_init, voxels=ICP_VOXELS[:2])
        candidates.append((alignment_score(cam_b, cam_a, T), T, f"features {method.upper()} ({inliers} inliers)"))
    return candidates, len(pts_a)


def register_pair(cam_a, cam_b, K=None, verbose=True):
    """Estima T_a_b: a transformação que leva pontos da câmara b para a câmara a.

    1. (extra) Hipóteses da mesa (table_hypotheses), se a mesa foi detetada nas duas imagens;
    2. Estimativa inicial: pares 3D SIFT -> várias hipóteses de RANSAC;
    3. ICP de cada hipótese; (extra) as que põem pontos em espaço livre são descartadas
       (is_plausible) e das restantes fica a de maior alignment_score;
    4. (extra) Se o resultado não for de confiança (is_confident), tenta-se outra vez com mais
       esforço: features ASIFT, teste de Lowe mais permissivo (mais associações) e mais hipóteses;
    5. ICP: a vencedora é refinada nas escalas finas.
    Saída: (T_a_b, matriz de informação, score, de confiança sim/não).
    """
    candidates = []
    if cam_a.table is not None and cam_b.table is not None:
        for T_init in table_hypotheses(cam_a.table, cam_b.table):
            # Já está a ~1 cm: só as escalas finas (as grosseiras podiam fazê-la 'deslizar')
            T, _, _ = multiscale_icp(cam_b, cam_a, T_init, voxels=ICP_VOXELS[2:])
            candidates.append((alignment_score(cam_b, cam_a, T), T, "mesa"))
    found, num_pairs = feature_candidates(cam_a, cam_b, "sift", ratio=LOWE_RATIO, max_hypotheses=4)
    candidates += found
    candidates = [c for c in candidates if is_plausible(cam_a, cam_b, K, c[1])]

    if not candidates or not choose_candidate(candidates)[1]:
        if verbose:
            best_score = max((c[0] for c in candidates), default=0.0)
            print(f"    {cam_a.name} <- {cam_b.name}: pouca confiança ({best_score:.3f}), "
                  f"nova tentativa com mais esforço (ASIFT)...")
        retry, num_pairs = feature_candidates(cam_a, cam_b, "asift", ratio=0.9, max_hypotheses=10)
        candidates += [c for c in retry if is_plausible(cam_a, cam_b, K, c[1])]
    if not candidates:
        # Nenhuma hipótese plausível: parte-se da identidade (o par vai ser marcado como falhado)
        T, _, _ = multiscale_icp(cam_b, cam_a, np.eye(4), voxels=ICP_VOXELS[:2])
        candidates = [(0.0, T, "identidade (nenhuma hipótese plausível)")]
    (score, T_coarse, origin), confident = choose_candidate(candidates)
    if confident and not is_confident(candidates):
        origin += ", confirmada pela mesa e pelas features"

    # Refinamento final da hipótese vencedora nas escalas finas do ICP
    T, result, information = multiscale_icp(cam_b, cam_a, T_coarse, voxels=ICP_VOXELS[2:])
    if verbose:
        flag = "" if confident else "  <-- FALHOU (não será usado)"
        print(f"    {cam_a.name} <- {cam_b.name}: {num_pairs} pares 3D, {len(candidates)} hipóteses, "
              f"escolhida: {origin} (score={score:.3f}); "
              f"ICP fitness={result.fitness:.3f}, rmse={result.inlier_rmse * 1000:.2f} mm{flag}")
    return T, information, score, confident


def register_scene(scene, loop_closure=True, verbose=True):
    """Regista todas as imagens da cena.

    1. Pré-processamento e features de cada imagem (prepare_cameras);
    2-3. Regista cada imagem com a seguinte (i -> i+1) e encadeia: T_0_(i+1) = T_0_i @ T_i_(i+1);
    (extra) pares sem confiança NÃO são usados; fecho de ciclo, câmaras isoladas e grafo de poses.
    Saída: (T_camera0_camera de cada câmara, câmaras registadas).
    """
    views = scene.views
    n = len(views)
    if verbose:
        print("Parte 1: registo")
    cameras = prepare_cameras(views, verbose)

    def pair(a, b):
        """Atalho: regista a imagem b com a imagem a."""
        return register_pair(cameras[a], cameras[b], scene.K, verbose)

    # Pares consecutivos. Cada aresta: (i, j, T_i_j, informação, de confiança, incerta)
    if verbose:
        print("  2-3. estimativa inicial e ICP de cada par de imagens vizinhas:")
    edges = []
    for i in range(n - 1):
        T, info, _, ok = pair(i, i + 1)
        edges.append((i, i + 1, T, info, ok, False))
    T_chain = chain_poses(edges)                      # encadeamento simples

    good = [e for e in edges if e[4]]                 # (extra) só as arestas de confiança
    if not loop_closure or n < 3:
        # Uma câmara ligada à 0 por um par falhado fica com a pose do encadeamento (o result.json
        # precisa de uma pose), mas NÃO entra na nuvem: com 2 imagens e o registo errado, a mesma
        # pilha aparecia duas vezes, em dois sítios da mesa (teste 16, cena 1)
        return T_chain, [p is not None for p in poses_from_edges(n, good)]

    good, poses = extra_edges(n, pair, good, views, verbose)
    # Câmaras ainda sem pose ficam com a pose do encadeamento (só para o result.json; não entram na nuvem)
    registered = [p is not None for p in poses]
    poses = [p if p is not None else T_chain[k] for k, p in enumerate(poses)]
    return optimize_pose_graph(poses, good), registered


def save_registration(folder, scene, T_cams, registered):
    """Grava o registo.json: a pose de cada câmara (grafo de poses) e as câmaras que
    entraram na nuvem agregada (False = registo falhado). É a entrada das outras partes.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    info = {
        "cena": str(Path(scene.folder).resolve()),
        "imagens": [v.name for v in scene.views],
        "T_camera0_camera": [np.round(T, 9).tolist() for T in T_cams],
        "camaras_registadas": [bool(ok) for ok in registered],
    }
    (folder / REGISTRATION_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_registration(folder):
    """Lê o registo.json gravado por save_registration.

    Saída: dicionário com a pasta da cena, os nomes das imagens, a pose de cada câmara, as câmaras
    registadas e a pasta das partes. As nuvens não são lidas aqui.
    """
    info = json.loads((Path(folder) / REGISTRATION_FILE).read_text(encoding="utf-8"))
    return {"scene_folder": Path(info["cena"]), "images": info["imagens"],
            "T_cams": [np.array(T) for T in info["T_camera0_camera"]],
            "registered": info["camaras_registadas"], "folder": Path(folder)}


def run_registration(scene_folder, output=None, loop_closure=True, verbose=True):
    """Parte 1 numa cena: regista as imagens, junta as nuvens (4. agregação) e grava o registo.json.

    Entrada: pasta da cena e pasta de saída (por omissão <cena>/partes).
    Saída: dicionário com a cena, as poses das câmaras, as câmaras registadas, a nuvem agregada (e a
    câmara de cada ponto).
    """
    t0 = time.time()
    scene = load_scene(scene_folder)
    output = Path(output) if output else Path(scene_folder) / OUTPUT_FOLDER
    if verbose:
        print(f"Cena {scene.folder.name}: {len(scene.views)} imagens")

    T_cams, registered = register_scene(scene, loop_closure=loop_closure, verbose=verbose)
    if verbose and not all(registered):
        left_out = [v.name for v, ok in zip(scene.views, registered) if not ok]
        print(f"  câmaras sem registo de confiança, deixadas de fora da nuvem: {', '.join(left_out)}")

    cloud, camera_index = aggregate(scene, T_cams, registered)
    save_registration(output, scene, T_cams, registered)
    if verbose:
        print(f"  4. nuvem agregada com {len(cloud.points)} pontos; registo gravado em "
              f"{output / REGISTRATION_FILE} ({time.time() - t0:.1f} s)")
    return {"scene": scene, "scene_folder": scene.folder, "images": [v.name for v in scene.views],
            "T_cams": T_cams, "registered": registered, "cloud": cloud,
            "camera_index": camera_index, "folder": output}


def main():
    """Lê os argumentos da linha de comandos e corre a Parte 1 (só quando se corre este ficheiro sozinho)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="pasta da cena")
    parser.add_argument("--no_loop_closure", action="store_true", help="registo só por encadeamento")
    parser.add_argument("--visualize", action="store_true", help="janelas 3D: nuvem agregada (cor dos píxeis e cor por câmara)")
    parser.add_argument("--open3d", action="store_true",
                        help="janelas antigas do Open3D (cores reais e uma cor por câmara)")
    args = parser.parse_args()

    r = run_registration(args.scene, loop_closure=not args.no_loop_closure)
    print(f"\nRegisto gravado em {r['folder']}. A seguir:\n"
          f"    python savi/extra_fotometria.py --scene {args.scene}\n"
          f"    python savi/parte2_segmentacao.py --scene {args.scene}")
    if args.visualize:
        show_windows(make_windows(figure_data(r)))
    if args.open3d:
        show_registration(r["scene"], r["T_cams"])


if __name__ == "__main__":
    main()
