#!/usr/bin/env python3
"""SAVIBoxes - Trabalho 1 de SAVI. EXTRA: CORREÇÃO FOTOMÉTRICA (exposição diferente em cada imagem).

=====================================================================================================
COMO CORRER
=====================================================================================================
    (a partir da pasta principal do trabalho, DEPOIS da Parte 1: python savi/parte1_registo.py --scene ...)
    python savi/extra_fotometria.py --scene test/cena_1
    python savi/extra_fotometria.py --scene test/cena_1 --visualize   # imagens e nuvem sem / com correção

Com a opção "Exposure changes" do simulador, cada imagem tem o seu ganho s e o seu desvio b, como numa
câmara com exposição automática (cor da imagem = s * cor + b): a mesma superfície aparece mais clara
numas imagens e mais escura noutras. Este extra, com o modelo da Parte 06 (I' = s * I + b):
    1. encontra os pontos vistos por mais do que uma câmara (pares de pontos que coincidem depois do
       registo da Parte 1);
    2. estima s e b de cada imagem em relação à imagem 000 por otimização (least_squares), com a
       imagem 000 fixa como referência;
    3. corrige as imagens a cores (grava <cena>/partes/fotometria/NNN.png);
    4. volta a colorir as nuvens com as imagens corrigidas: as Partes 2 e 3 passam a usar estas cores.
Grava <cena>/partes/fotometria.json com o s e o b de cada imagem e o erro de cor nos pontos comuns,
antes e depois da correção. A comparação com o campo 'photometry' do ground_truth.json e o efeito na
segmentação por cor do Nível 4 estão no avaliador.py (secção 5).

Ordem das partes (o run_test.py corre-as todas, cena a cena):
    parte1_registo.py      -> <cena>/partes/registo.json   (pose de cada câmara)
    extra_fotometria.py    -> <cena>/partes/fotometria.json e fotometria/ (s e b, imagens corrigidas)
    parte2_segmentacao.py  -> <cena>/partes/objetos.ply, mesa.json, segmentacao.npz (estima o n.º de caixas)
    parte3_ajuste.py       -> result.json e <cena>/partes/ajuste.npz
    avaliador.py           -> pede o n.º real de caixas e compara com o ground truth (se a cena o tiver)

=====================================================================================================
ORGANIZAÇÃO DESTE FICHEIRO (de onde vem cada passo)
=====================================================================================================
    1. Pontos comuns        nuvens no referencial da câmara 0, voxel_down_sample (Parte 07, Ex. 4) e pares
                            de pontos a menos de 1 cm com o evaluate_registration do Open3D (o
                            correspondence_set do ICP, Parte 08, Ex. 2)
    2. Estimar s e b        objective_function + least_squares (Parte 06, Ex. 1 e 2): um erro por par de
                            pontos e canal (Ex. 2.3), imagem 000 fixa (Ex. 2.2: sem referência a solução
                            degenera em imagens pretas, Ex. 2.1), uma loss robusta para os pares errados
                            (Ex. 5; aqui 'cauchy', ver estimate_exposures) e bounds para os valores
                            possíveis (Ex. 4.3)
    3. Corrigir as imagens  I' = s * I + b, limitado a [0, 255] com np.minimum / np.maximum (Parte 06, Ex. 1)
    4. Recolorir as nuvens  cada ponto da nuvem é projetado com K no seu pixel da imagem corrigida
                            (u = fx x / z + cx, v = fy y / z + cy). NÃO está numa Parte: é o inverso da
                            passagem da imagem de profundidade para a nuvem (simulador, depth_to_point_cloud)
                            e é a mesma projeção do cloud_to_xyz_image do comum.py
    5. Visualização         mosaico das imagens sem / com correção (como o buildMosaic do Ex. 2) e a
                            nuvem agregada sem / com correção
    6. Programa principal   run_photometry() e main()
"""

import argparse
import copy
import dataclasses
import json
from functools import partial
from pathlib import Path

import cv2                                   # OpenCV: gravar as imagens corrigidas e o mosaico
import numpy as np
import open3d as o3d                         # nuvens de pontos: voxel e pares de pontos (correspondências)
from scipy.optimize import least_squares     # otimização (Parte 06)

from comum import OUTPUT_FOLDER, load_scene
from visualizacao import colors_uint8, data_view, image_item, points_item, scene_view, show_windows

# A Parte 1 importa este ficheiro (exposure_corrected, para comparar cores no registo); por isso, aqui, a
# Parte 1 só é importada dentro das funções que precisam dela (import parte1_registo as p1).

PHOTOMETRY_FILE = "fotometria.json"     # s e b estimados de cada imagem
CORRECTED_FOLDER = "fotometria"         # imagens a cores corrigidas

PAIR_VOXEL = 0.01        # 1. m: voxel das nuvens antes de procurar os pares (Parte 07)
PAIR_DISTANCE = 0.01     # 1. m: dois pontos a menos disto são o mesmo ponto visto por duas câmaras
MAX_PAIRS = 3000         # 1. pares por par de câmaras (menos dados, como o scaling_factor do Ex. 2)
SATURATION = (0.02, 0.98)  # 1. cores fora disto foram cortadas a [0, 1] e não seguem o modelo s * cor + b
S_BOUNDS = (0.1, 10.0)   # 2. valores possíveis de s (bounds do least_squares, Parte 06, Ex. 4.3)
B_BOUNDS = (-1.0, 1.0)   # 2. valores possíveis de b (as cores estão entre 0 e 1)
LOSS = "cauchy"          # 2. loss robusta do least_squares (Parte 06, Ex. 5; ver estimate_exposures)
F_SCALE = 0.01           # 2. diferença de cor a partir da qual a loss reduz o peso de um par (~ ruído de cor)


# #####################################################################################################
# 1. PONTOS VISTOS POR MAIS DO QUE UMA CÂMARA
# #####################################################################################################
def unsaturated(colors):
    """Máscara das cores que não foram cortadas a 0 ou a 1.

    Uma cor cortada já não é s * cor + b (o simulador e a câmara cortam a cor a [0, 1], tal como o
    np.minimum / np.maximum da Parte 06, Ex. 1), por isso esses pontos ficam fora da estimação. É a
    mesma ideia da erosão da máscara da Parte 06, Ex. 3: tirar os pixeis onde o modelo não vale.
    """
    return np.all((colors > SATURATION[0]) & (colors < SATURATION[1]), axis=1)


def color_pairs(cloud_i, cloud_j, rng, distance=PAIR_DISTANCE, max_pairs=MAX_PAIRS):
    """Cores dos pontos vistos pelas duas câmaras: pares de pontos a menos de 'distance' metros.

    As duas nuvens já estão no referencial da câmara 0, por isso as correspondências são as do ICP
    (Parte 08) com a identidade: evaluate_registration devolve, para cada ponto de cloud_i, o vizinho
    mais próximo em cloud_j, se estiver a menos de 'distance'.
    Saída: (cores em cloud_i, cores em cloud_j), uma linha por par, até max_pairs pares.
    """
    result = o3d.pipelines.registration.evaluate_registration(cloud_i, cloud_j, distance, np.eye(4))
    pairs = np.asarray(result.correspondence_set)
    if len(pairs) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    colors_i = np.asarray(cloud_i.colors)[pairs[:, 0]]
    colors_j = np.asarray(cloud_j.colors)[pairs[:, 1]]
    keep = unsaturated(colors_i) & unsaturated(colors_j)
    colors_i, colors_j = colors_i[keep], colors_j[keep]
    if len(colors_i) > max_pairs:
        chosen = rng.choice(len(colors_i), max_pairs, replace=False)
        colors_i, colors_j = colors_i[chosen], colors_j[chosen]
    return colors_i, colors_j


def common_points(scene, T_camera0_cameras, registered):
    """Pares de pontos vistos por duas câmaras, para todos os pares de câmaras registadas.

    Cada nuvem é levada para o referencial da câmara 0 (pose da Parte 1) e reduzida com um voxel de 1 cm.
    Saída: lista de (i, j, cores em i, cores em j), i < j índices das imagens.
    """
    rng = np.random.default_rng(0)               # sempre os mesmos pares (resultados repetíveis)
    clouds = {}
    for k, (view, T) in enumerate(zip(scene.views, T_camera0_cameras)):
        if registered[k]:
            cloud = copy.deepcopy(view.cloud)        # cópia, para não alterar a nuvem original
            cloud.transform(T)                       # referencial da câmara 0
            clouds[k] = cloud.voxel_down_sample(PAIR_VOXEL)
    pairs = []
    for i in clouds:
        for j in clouds:
            if i < j:
                colors_i, colors_j = color_pairs(clouds[i], clouds[j], rng)
                if len(colors_i):
                    pairs.append((i, j, colors_i, colors_j))
    return pairs


# #####################################################################################################
# 2. ESTIMAR s E b DE CADA IMAGEM EM RELAÇÃO À IMAGEM 000 (Parte 06)
# #####################################################################################################
def objective_function(params, data):
    """Um erro de similaridade por par de pontos e por canal (Parte 06, Ex. 2):

        e = (s_i * cor_i + b_i) - (s_j * cor_j + b_j)

    params = [s, b] de cada imagem de data['images'] (sem a 000). A imagem 000 é a referência e fica
    fixa com s = 1 e b = 0: se também mudasse, todas as imagens ficavam iguais e pretas (Ex. 2.1).
    """
    s, b = np.ones(data["num_images"]), np.zeros(data["num_images"])
    s[data["images"]], b[data["images"]] = params[0::2], params[1::2]
    errors = [((s[i] * colors_i + b[i]) - (s[j] * colors_j + b[j])).ravel()
              for i, j, colors_i, colors_j in data["pairs"]]
    return np.concatenate(errors)


def linked_to_reference(pairs, reference=0):
    """Imagens ligadas à imagem 000 por pares de pontos (diretamente ou através de outras imagens).

    Uma imagem que não se sobrepõe à 000 também é estimada, através das imagens que se sobrepõem às
    duas. Um grupo de imagens sem ligação à 000 não tem referência fixa e degeneraria (Ex. 2.1).
    """
    linked, grew = {reference}, True
    while grew:
        grew = False
        for i, j, _, _ in pairs:
            if (i in linked) != (j in linked):
                linked |= {i, j}
                grew = True
    return linked


def estimate_exposures(pairs, num_images, reference=0):
    """s e b de cada imagem (cor_000 = s * cor + b) por least_squares, com a imagem 'reference' fixa.

    Pares errados: pares a menos de 1 cm que não são o mesmo ponto (arestas entre caixa e mesa, sombras,
    cores misturadas pelo desalinhamento cor/profundidade). Como na Parte 06, Ex. 5, usa-se uma loss que
    reduz o peso dos erros grandes. Com o soft_l1 do Ex. 5 o s ainda ficava ~3% abaixo do verdadeiro
    (erro médio de s 0.030 na cena level_4 com Exposure changes; 0.138 sem loss robusta); com 'cauchy',
    que reduz ainda mais o peso dos erros grandes, o erro médio de s desce para 0.007.
    Entrada: os pares de common_points e o número de imagens.
    Saída: (s, b, estimada) com um valor por imagem; estimada[k] = False para as imagens sem ligação à
    referência (ficam com s = 1 e b = 0, sem correção).
    """
    linked = linked_to_reference(pairs, reference)
    images = sorted(linked - {reference})
    s, b = np.ones(num_images), np.zeros(num_images)
    estimated = np.zeros(num_images, bool)
    estimated[reference] = True
    if not images:
        return s, b, estimated
    data = {"pairs": [p for p in pairs if p[0] in linked and p[1] in linked],
            "images": images, "num_images": num_images}
    x0 = [1.0, 0.0] * len(images)                # s = 1 e b = 0: imagem sem correção (Ex. 1)
    lower = [S_BOUNDS[0], B_BOUNDS[0]] * len(images)
    upper = [S_BOUNDS[1], B_BOUNDS[1]] * len(images)
    result = least_squares(partial(objective_function, data=data), x0, bounds=(lower, upper),
                           loss=LOSS, f_scale=F_SCALE)
    s[images], b[images] = result.x[0::2], result.x[1::2]
    estimated[images] = True
    return s, b, estimated


def exposure_corrected(colors, colors_target, close):
    """Cores de uma nuvem com a exposição corrigida para a de outra (usado no registo, Parte 1).

    É a Parte 06, Ex. 2.2, tal e qual: a nuvem alvo é a referência fixa e só a outra é corrigida
    (2 parâmetros), com os pares de pontos que coincidem em geometria ('close').
    Sem isto, comparar cores de imagens com exposições diferentes dava score ~0 a TODAS as hipóteses
    de registo de um par, e a certa deixava de se distinguir da errada (mesa rodada 180°).
    """
    ok = close & unsaturated(colors) & unsaturated(colors_target)
    if not ok.any():
        return colors
    s, b, _ = estimate_exposures([(0, 1, colors_target[ok], colors[ok])], 2)
    return np.clip(s[1] * colors + b[1], 0.0, 1.0)


def color_error(pairs, s, b):
    """Diferença média de cor entre os pares de pontos, depois de aplicar s e b a cada imagem
    (com s = 1 e b = 0 é o erro sem correção). É a média dos erros da objective_function."""
    errors = [np.abs((s[i] * ci + b[i]) - (s[j] * cj + b[j])).mean(axis=1) for i, j, ci, cj in pairs]
    return float(np.concatenate(errors).mean()) if errors else float("nan")


# #####################################################################################################
# 3. CORRIGIR AS IMAGENS A CORES
# #####################################################################################################
def correct_image(rgb, s, b):
    """I' = s * I + b (Parte 06, Ex. 1), com I entre 0 e 1 e o resultado limitado a [0, 1]."""
    corrected = s * rgb.astype(float) / 255.0 + b
    corrected = np.minimum(corrected, 1.0)
    corrected = np.maximum(corrected, 0.0)
    return (corrected * 255 + 0.5).astype(np.uint8)


def save_images(scene, folder):
    """Grava as imagens a cores (já corrigidas) em folder/NNN.png."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for view in scene.views:
        bgr = view.rgb[..., ::-1]                                   # o OpenCV grava em BGR
        # cv2.imwrite falha no Windows com caminhos com acentos: codifica-se a imagem e grava-se com numpy
        cv2.imencode(".png", bgr)[1].tofile(str(folder / f"{view.name}.png"))


# #####################################################################################################
# 4. VOLTAR A COLORIR AS NUVENS COM AS IMAGENS CORRIGIDAS
# #####################################################################################################
def recolor_cloud(cloud, rgb, K):
    """Nuvem com a cor do seu pixel na imagem rgb.

    Cada ponto da nuvem (referencial da sua câmara) foi medido num pixel da imagem; projetando-o com K
    volta-se a esse pixel: u = fx * x / z + cx, v = fy * y / z + cy (como no cloud_to_xyz_image do
    comum.py). Os pontos que não caem na imagem ficam com a cor que tinham.
    """
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors).copy()
    height, width = rgb.shape[:2]
    z = points[:, 2]
    valid = z > 0
    u = np.zeros(len(points), int)
    v = np.zeros(len(points), int)
    u[valid] = np.round(K[0, 0] * points[valid, 0] / z[valid] + K[0, 2]).astype(int)
    v[valid] = np.round(K[1, 1] * points[valid, 1] / z[valid] + K[1, 2]).astype(int)
    inside = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    colors[inside] = rgb[v[inside], u[inside]] / 255.0
    recolored = copy.deepcopy(cloud)
    recolored.colors = o3d.utility.Vector3dVector(colors)
    return recolored


def corrected_scene(scene, rows):
    """Cena com as imagens corrigidas (3.) e as nuvens coloridas de novo com elas (4.).

    rows: lista de {imagem, s, b} (a de run_photometry ou a do fotometria.json). As imagens que não
    estão em rows ficam como estavam. A cena original não é alterada.
    """
    exposure = {row["imagem"]: (row["s"], row["b"]) for row in rows}
    views = []
    for view in scene.views:
        if view.name in exposure:
            rgb = correct_image(view.rgb, *exposure[view.name])
            view = dataclasses.replace(view, rgb=rgb, cloud=recolor_cloud(view.cloud, rgb, scene.K))
        views.append(view)
    return dataclasses.replace(scene, views=views)


def load_corrected_scene(scene, parts_folder):
    """A cena com a correção do fotometria.json, se o extra já correu nesta cena (senão, a própria cena).
    Usado pelas Partes 2 e 3 quando correm sozinhas (lêem o registo e a fotometria gravados)."""
    path = Path(parts_folder) / PHOTOMETRY_FILE
    if not path.exists():
        return scene
    return corrected_scene(scene, json.loads(path.read_text(encoding="utf-8"))["imagens"])


# #####################################################################################################
# 5. VISUALIZAÇÃO
# #####################################################################################################
def mosaic(images, labels, width=240):
    """Imagens lado a lado (4 por linha), reduzidas para 'width' pixeis de largura, cada uma com a sua
    legenda (como o mosaico da Parte 06, Ex. 2, para ver se as imagens ficam com a mesma cor)."""
    tiles = []
    for image, label in zip(images, labels):
        tile = cv2.resize(image, (width, int(image.shape[0] * width / image.shape[1])), interpolation=cv2.INTER_AREA)
        tile = np.ascontiguousarray(tile)
        cv2.rectangle(tile, (0, 0), (width - 1, 24), (255, 255, 255), -1)     # faixa branca da legenda
        cv2.putText(tile, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
    return np.vstack(rows)


def figure_data(r1, rx):
    """Dados das janelas do extra (r1 = Parte 1, rx = run_photometry ou o fotometria.json): as imagens e a
    nuvem agregada inteira sem correção e com correção, e o s e o b de cada imagem."""
    import parte1_registo as p1
    raw_scene = load_scene(r1["scene_folder"])
    rows = rx["imagens"]
    scene = corrected_scene(raw_scene, rows)
    raw, _ = p1.aggregate(raw_scene, r1["T_cams"], r1["registered"])
    corrected, _ = p1.aggregate(scene, r1["T_cams"], r1["registered"])
    exposure = {row["imagem"]: row for row in rows}
    labels = [f"{v.name}: s={exposure[v.name]['s']:.2f} b={exposure[v.name]['b']:+.2f}" if v.name in exposure
              else f"{v.name}: sem estimativa" for v in scene.views]
    images = np.vstack([mosaic([v.rgb for v in raw_scene.views], [f"{v.name} (original)" for v in raw_scene.views]),
                        np.full((12, 4 * 240, 3), 255, np.uint8),
                        mosaic([v.rgb for v in scene.views], labels)])
    view = scene_view(r1["T_cams"], r1["registered"])
    return {"parte": "extra_fotometria", "cena": Path(r1["scene_folder"]).name,
            "images": images, "points": np.asarray(raw.points, np.float32),
            "colors_raw": colors_uint8(raw.colors), "colors_corrected": colors_uint8(corrected.colors),
            "error_before": rx["erro_cor_antes"], "error_after": rx["erro_cor_depois"],
            **{f"view_{k}": v for k, v in view.items()}}


def make_windows(data):
    """Janelas do extra: o mosaico das imagens (em cima as originais, em baixo as corrigidas, com o s e o
    b de cada uma) e a nuvem agregada com as cores originais e com as cores corrigidas."""
    title = f"Extra - {data['cena']}: correção fotométrica, cor_000 = s * cor + b"
    errors = (f"erro de cor nos pontos comuns: {float(data['error_before']):.3f} sem correção, "
              f"{float(data['error_after']):.3f} com correção")
    view = data_view(data)
    return [{"name": "_imagens", "title": f"{title} - imagens: em cima originais, em baixo corrigidas",
             "items": [image_item(data["images"])]},
            {"name": "_sem_correcao", "title": f"{title} - nuvem SEM correção ({errors})",
             "items": [points_item(data["points"], data["colors_raw"])], "view": view},
            {"name": "_com_correcao", "title": f"{title} - nuvem COM correção ({errors})",
             "items": [points_item(data["points"], data["colors_corrected"])], "view": view}]


# #####################################################################################################
# 6. PROGRAMA PRINCIPAL
# #####################################################################################################
def run_photometry(r1, verbose=True):
    """Extra numa cena: s e b de cada imagem, imagens corrigidas, nuvens coloridas de novo e fotometria.json.

    Entrada: o resultado da Parte 1 (run_registration ou load_registration).
    Atualiza r1: r1["scene"] passa a ter as imagens e as nuvens corrigidas e r1["cloud"] (nuvem agregada)
    é juntada outra vez com elas; é esta nuvem que a Parte 2 segmenta.
    Saída: dicionário com a lista de {imagem, s, b} e o erro de cor nos pontos comuns antes e depois.
    """
    import parte1_registo as p1
    scene = load_scene(r1["scene_folder"])       # cores originais (r1["scene"] pode já estar corrigida)
    names = [v.name for v in scene.views]
    pairs = common_points(scene, r1["T_cams"], r1["registered"])        # 1.
    s, b, estimated = estimate_exposures(pairs, len(scene.views))       # 2.
    rows = [{"imagem": names[k], "s": float(s[k]), "b": float(b[k])} for k in range(len(names)) if estimated[k]]
    info = {"imagens": rows, "erro_cor_antes": color_error(pairs, np.ones(len(s)), np.zeros(len(b))),
            "erro_cor_depois": color_error(pairs, s, b)}

    corrected = corrected_scene(scene, rows)                            # 3. e 4.
    folder = Path(r1["folder"])
    save_images(corrected, folder / CORRECTED_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / PHOTOMETRY_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
    r1["scene"] = corrected
    r1["cloud"], r1["camera_index"] = p1.aggregate(corrected, r1["T_cams"], r1["registered"])

    if verbose:
        print("Extra - fotometria (cor_000 = s * cor + b):")
        for row in rows:
            print(f"  imagem {row['imagem']}: s = {row['s']:.3f}, b = {row['b']:+.3f}")
        for k, name in enumerate(names):
            if not estimated[k]:
                print(f"  imagem {name}: sem pontos comuns ligados à 000, não foi corrigida")
        print(f"  erro de cor nos pontos comuns ({sum(len(p[2]) for p in pairs)} pares): "
              f"{info['erro_cor_antes']:.4f} sem correção -> {info['erro_cor_depois']:.4f} com correção")
        print(f"  imagens corrigidas em {folder / CORRECTED_FOLDER}")
    return info


def main():
    """Lê os argumentos da linha de comandos e corre o extra."""
    import parte1_registo as p1
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="pasta da cena (já com a Parte 1 feita)")
    parser.add_argument("--visualize", action="store_true", help="janelas: imagens e nuvem agregada sem e com correção")
    args = parser.parse_args()

    r1 = p1.load_registration(Path(args.scene) / OUTPUT_FOLDER)
    rx = run_photometry(r1)
    if args.visualize:
        show_windows(make_windows(figure_data(r1, rx)))


if __name__ == "__main__":
    main()
