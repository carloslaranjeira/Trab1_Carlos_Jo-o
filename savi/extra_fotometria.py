#!/usr/bin/env python3
"""SAVIBoxes - Trabalho 1 de SAVI. EXTRA: CORREÇÃO FOTOMÉTRICA (exposição diferente em cada imagem).

=====================================================================================================
COMO CORRER
=====================================================================================================
    (a partir da pasta principal do trabalho, DEPOIS da Parte 1: python savi/parte1_registo.py --scene ...)
    python savi/extra_fotometria.py --scene test/cena_1
    python savi/extra_fotometria.py --scene test/cena_1 --visualize   # janelas 3D: nuvem sem / com correção

Com a opção "Exposure changes" do simulador, cada imagem tem o seu ganho s e o seu desvio b, como numa
câmara com exposição automática (cor da imagem = s * cor + b): a mesma superfície aparece mais clara
numas imagens e mais escura noutras. Este extra (Parte 06):
    1. estima s e b de cada imagem em relação à imagem 000, a partir dos pontos vistos por mais do que
       uma câmara (pontos que coincidem na nuvem agregada), por mínimos quadrados robustos;
    2. corrige as imagens a cores (grava <cena>/partes/fotometria/NNN.png) e volta a colorir as nuvens
       (a nuvem agregada da Parte 1 já é construída com as cores corrigidas);
    3. compara os valores estimados com os do campo 'photometry' do ground_truth.json (se existir).
A mesma correção é usada no registo (Parte 1), para comparar cores de imagens com exposições diferentes,
e chega corrigida à segmentação por cor (Parte 2, Nível 4).
Grava <cena>/partes/fotometria.json com s e b de cada imagem (estimados e, se houver, verdadeiros).

Ordem das partes (o run_test.py corre-as todas, cena a cena):
    parte1_registo.py      -> <cena>/partes/registo.json   (pose de cada câmara)
    extra_fotometria.py    -> <cena>/partes/fotometria.json (ganho e desvio de cada imagem)
    parte2_segmentacao.py  -> <cena>/partes/objetos.ply, mesa.json, segmentacao.npz (estima o n.º de caixas)
    parte3_ajuste.py       -> result.json e <cena>/partes/ajuste.npz
    avaliador.py           -> pede o n.º real de caixas e compara com o ground truth (se a cena o tiver)

=====================================================================================================
ORGANIZAÇÃO DESTE FICHEIRO
=====================================================================================================
    1. Modelo da exposição  estimar s e b (exposure_fit, match_exposure) e corrigir cores (exposure_corrected)
    2. Correção das imagens s e b de cada imagem, valores verdadeiros, imagens corrigidas
    3. Visualização         janelas 3D: nuvem agregada sem correção e com correção
    4. Programa principal   run_photometry() e main()
"""

import argparse
import json
from pathlib import Path

import cv2                                   # OpenCV: gravar as imagens corrigidas
import numpy as np
from scipy.spatial import cKDTree            # procura rápida do vizinho mais próximo

from comum import OUTPUT_FOLDER, load_scene
from visualizacao import colors_uint8, data_view, points_item, scene_view, show_windows

# A Parte 1 importa este ficheiro (exposure_corrected, match_exposure); por isso, aqui, a Parte 1 só é
# importada dentro das funções que precisam dela (import parte1_registo as p1).


# #####################################################################################################
# 1. MODELO DA EXPOSIÇÃO: cor' = s * cor + b
# #####################################################################################################
def exposure_fit(colors, colors_target, close, min_points=50):
    """Ganho s e desvio b que levam a exposição de uma imagem à de outra: cor' = s * cor + b.

    Com exposição automática, a mesma superfície pode aparecer mais clara ou mais escura em cada
    imagem (no teste 10, uma imagem tinha s = 0.66 e b = -0.06). s e b estimam-se por mínimos
    quadrados com os pares de pontos que coincidem em geometria ('close'), em duas passagens: a
    segunda sem os 30% de pares com maior resíduo (arestas, sombras, pares de objetos diferentes).
    Sem pares suficientes: s = 1, b = 0 (não se corrige).
    """
    if close.sum() < min_points:
        return 1.0, 0.0
    x, y = colors[close].ravel(), colors_target[close].ravel()
    keep = np.ones(len(x), bool)
    for _ in range(2):
        A = np.stack([x[keep], np.ones(keep.sum())], axis=1)
        s, b = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        residual = np.abs(y - (s * x + b))
        keep = residual <= np.percentile(residual, 70)
    return float(np.clip(s, 0.5, 2.0)), float(np.clip(b, -0.2, 0.2))


def exposure_corrected(colors, colors_target, close):
    """Cores de uma imagem com a exposição corrigida para a de outra (ver exposure_fit).

    Sem isto, comparar cores diretamente dava score ~0 a TODAS as hipóteses de um par com exposições
    diferentes, e a certa deixava de se distinguir da errada (mesa rodada 180°).
    """
    s, b = exposure_fit(colors, colors_target, close)
    return np.clip(s * colors + b, 0.0, 1.0)


def match_exposure(cloud, reference, distance=0.01):
    """Ganho s e desvio b que levam a exposição de uma nuvem à de outra nuvem já no mesmo referencial.

    Usado pela Parte 1 ao juntar as nuvens (aggregate): cada nuvem passa a ter a exposição das anteriores.
    Os pares de pontos são os vizinhos mais próximos a menos de 'distance' metros (nuvens reduzidas a
    1 cm). A cromaticidade (chroma_features da Parte 2) cancela o ganho s mas não o desvio b, e com cores
    de exposições diferentes duas caixas de cores parecidas (laranja e castanho) deixavam de se separar.
    """
    small, ref = cloud.voxel_down_sample(0.01), reference.voxel_down_sample(0.01)
    dist, idx = cKDTree(np.asarray(ref.points)).query(np.asarray(small.points), k=1)
    return exposure_fit(np.asarray(small.colors), np.asarray(ref.colors)[idx], dist < distance)


# #####################################################################################################
# 2. CORREÇÃO DE CADA IMAGEM EM RELAÇÃO À IMAGEM 000
# #####################################################################################################
PHOTOMETRY_FILE = "fotometria.json"     # s e b de cada imagem (estimados e, se houver, verdadeiros)
CORRECTED_FOLDER = "fotometria"         # imagens a cores corrigidas


def true_exposures(ground_truth):
    """s e b VERDADEIROS de cada imagem em relação à imagem 000 (só para avaliar).

    O ground_truth.json dá, para cada imagem k, cor_k = s_k * cor + b_k. Tirando a cor verdadeira:
        cor_0 = s_0 / s_k * (cor_k - b_k) + b_0 = s_rel * cor_k + b_rel
    com s_rel = s_0 / s_k e b_rel = b_0 - s_0 * b_k / s_k. É isto que se estima, porque a imagem 000 é a
    referência. Saída: {imagem: (s_rel, b_rel)}, ou {} se não houver ground truth.
    """
    if not ground_truth or "photometry" not in ground_truth:
        return {}
    photometry = {p["image"]: (p["s"], p["b"]) for p in ground_truth["photometry"]}
    s0, b0 = photometry[min(photometry)]
    return {name: (s0 / s, b0 - s0 * b / s) for name, (s, b) in photometry.items()}


def correct_images(scene, exposures, folder):
    """Corrige as imagens a cores com o s e o b estimados (cor' = s * cor + b) e grava-as em folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    views = {v.name: v for v in scene.views}
    for name, s, b in exposures:
        corrected = np.clip(s * views[name].rgb.astype(float) / 255.0 + b, 0.0, 1.0)
        bgr = (corrected[..., ::-1] * 255 + 0.5).astype(np.uint8)      # o OpenCV grava em BGR
        # cv2.imwrite falha no Windows com caminhos com acentos: codifica-se a imagem e grava-se com numpy
        cv2.imencode(".png", bgr)[1].tofile(str(folder / f"{name}.png"))


# #####################################################################################################
# 3. VISUALIZAÇÃO
# #####################################################################################################
def figure_data(r1, rx):
    """Dados das janelas do extra (r1 = Parte 1, rx = run_photometry): a nuvem agregada inteira com as cores
    sem correção e com correção, e o s e o b de cada imagem."""
    import parte1_registo as p1
    scene = r1["scene"] if r1.get("scene") is not None else load_scene(r1["scene_folder"])
    raw, _ = p1.aggregate(scene, r1["T_cams"], r1["registered"], photometric=False)
    corrected = r1.get("cloud")
    if corrected is None or len(corrected.points) != len(raw.points):
        corrected, _ = p1.aggregate(scene, r1["T_cams"], r1["registered"])
    rows = rx["photometry"]
    nan = float("nan")
    view = scene_view(r1["T_cams"], r1["registered"])
    return {"parte": "extra_fotometria", "cena": Path(r1["scene_folder"]).name,
            "points": np.asarray(raw.points, np.float32), "colors_raw": colors_uint8(raw.colors),
            "colors_corrected": colors_uint8(corrected.colors),
            "images": np.array([row["imagem"] for row in rows]), "s": np.array([row["s"] for row in rows]),
            "b": np.array([row["b"] for row in rows]),
            "s_true": np.array([nan if row["s_verdadeiro"] is None else row["s_verdadeiro"] for row in rows]),
            "b_true": np.array([nan if row["b_verdadeiro"] is None else row["b_verdadeiro"] for row in rows]),
            **{f"view_{k}": v for k, v in view.items()}}


def make_windows(data):
    """Janelas do extra: a nuvem agregada com as cores originais de cada imagem e com a exposição de todas
    as imagens corrigida para a da imagem 000 (cada uma na sua janela; s e b de cada imagem no título)."""
    values = []
    for name, s, b, st, bt in zip(data["images"], data["s"], data["b"], data["s_true"], data["b_true"]):
        value = f"{name}: s={s:.3f} b={b:+.3f}"
        if not np.isnan(st):
            value += f" (verd. {st:.3f} {bt:+.3f})"
        values.append(value)
    title = f"Extra - {data['cena']}: correção fotométrica, cor_000 = s * cor + b"
    view = data_view(data)
    return [{"name": "_sem_correcao", "title": f"{title} - SEM correção (cada imagem com a sua exposição)",
             "items": [points_item(data["points"], data["colors_raw"])], "view": view},
            {"name": "_com_correcao", "title": f"{title} - COM correção: {'; '.join(values)}",
             "items": [points_item(data["points"], data["colors_corrected"])], "view": view}]


# #####################################################################################################
# 4. PROGRAMA PRINCIPAL
# #####################################################################################################
def run_photometry(r1, verbose=True):
    """Extra numa cena: s e b de cada imagem, comparação com o ground truth, imagens corrigidas e
    fotometria.json.

    Entrada: o resultado da Parte 1. Com run_registration, s e b já foram estimados ao juntar as nuvens
    (aggregate); com load_registration, as nuvens voltam a ser juntadas para os estimar.
    Saída: dicionário com a lista de {imagem, s, b, s_verdadeiro, b_verdadeiro}.
    """
    import parte1_registo as p1
    scene = r1["scene"] if r1.get("scene") is not None else load_scene(r1["scene_folder"])
    exposures = r1.get("exposures")
    if exposures is None:
        exposures = []
        p1.aggregate(scene, r1["T_cams"], r1["registered"], exposures=exposures)
    truth = true_exposures(scene.ground_truth)
    folder = Path(r1["folder"])
    correct_images(scene, exposures, folder / CORRECTED_FOLDER)

    rows = [{"imagem": name, "s": float(s), "b": float(b),
             "s_verdadeiro": truth[name][0] if name in truth else None,
             "b_verdadeiro": truth[name][1] if name in truth else None} for name, s, b in exposures]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / PHOTOMETRY_FILE).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if verbose:
        print("Extra - fotometria (cor_000 = s * cor + b):")
        for row in rows:
            line = f"  imagem {row['imagem']}: s = {row['s']:.3f}, b = {row['b']:+.3f}"
            if row["s_verdadeiro"] is not None:
                line += f"   (verdadeiros: s = {row['s_verdadeiro']:.3f}, b = {row['b_verdadeiro']:+.3f})"
            print(line)
        estimated = {row["imagem"] for row in rows}
        for v in scene.views:
            if v.name not in estimated:
                print(f"  imagem {v.name}: sem registo de confiança, não foi estimada")
        print(f"  imagens corrigidas em {folder / CORRECTED_FOLDER}")
    return {"photometry": rows}


def main():
    """Lê os argumentos da linha de comandos e corre o extra."""
    import parte1_registo as p1
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="pasta da cena (já com a Parte 1 feita)")
    parser.add_argument("--visualize", action="store_true", help="janelas 3D: nuvem agregada sem e com correção")
    args = parser.parse_args()

    r1 = p1.load_registration(Path(args.scene) / OUTPUT_FOLDER)
    rx = run_photometry(r1)
    if args.visualize:
        show_windows(make_windows(figure_data(r1, rx)))


if __name__ == "__main__":
    main()
