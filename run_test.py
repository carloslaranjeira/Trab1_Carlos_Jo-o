#!/usr/bin/env python3
"""Dia da apresentação: corre o sistema em TODAS as cenas de uma pasta, uma de cada vez.

Utilização:
    python run_test.py --download <link do repositório GitHub>   # copia as cenas do GitHub para test/ e corre
    python run_test.py              # corre todas as cenas que já estão em test/
    python run_test.py --git        # e faz git add + commit + push depois de cada cena
    python run_test.py --redo       # volta a correr cenas que já têm resultado
    python run_test.py --sem_janelas   # não abre janelas (só grava as figuras em results_test/<cena>/)

Corre as cenas UMA DE CADA VEZ: em cada cena, os 4 ficheiros (savi/) um depois do outro e o avaliador,
e no fim mostra a janela da cena; só depois passa à cena seguinte:
    parte1_registo.py      -> test/<cena>/partes/registo.json   (pose de cada câmara)
    extra_fotometria.py    -> test/<cena>/partes/fotometria.json (exposição de cada imagem)
    parte2_segmentacao.py  -> test/<cena>/partes/objetos.ply, mesa.json, segmentacao.npz
    parte3_ajuste.py       -> results_test/<cena>/result.json (e top_view.png)
    avaliador.py           -> número de caixas e erros em relação ao ground truth (se a cena o tiver)
As 4 partes NÃO usam o número de caixas (a Parte 2 estima-o). O número real só é pedido no avaliador,
no fim de cada cena, para o comparar com o número de caixas encontradas; o de uma cena que esteja em
test/num_boxes.json não é perguntado.

Janelas: UMA janela 3D do Open3D por cena, com as vistas de todas as partes (como a figura da nuvem
agregada do enunciado); as setas <- e -> do teclado mudam de vista e o título diz qual é. A janela
de uma cena abre quando as 4 partes acabam (antes da pergunta do avaliador), sem parar o programa: as janelas são abertas por um
segundo processo (python run_test.py --janelas ...), por isso pode ver-se, por exemplo, a cena 3 sem
fechar as janelas das cenas 1 e 2, e o programa continua enquanto estão abertas. Cada vista é gravada
também em PNG em results_test/<cena>/ (parte1_registo_cor_pixeis.png, parte1_registo_cor_camaras.png, ...).

Pensado para não perder resultados:
    - um erro numa cena não pára as outras (fica registado e passa-se à seguinte);
    - cenas que já têm result.json são saltadas, por isso pode voltar-se a correr o script;
    - com --git, cada resultado é enviado para o GitHub logo que fica pronto.
"""

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).resolve().parent         # pasta principal do trabalho
# O código está na pasta savi/: junta-a aos sítios onde o Python procura módulos
sys.path.insert(0, str(FOLDER / "savi"))

WINDOWS_FOLDER = FOLDER / "test" / ".janelas"    # pedidos de janelas para o processo das janelas


def natural_key(path):
    """Chave de ordenação 'natural': os números no nome são comparados como números.

    'cena_10' -> ['cena_', 10, ''], por isso fica depois de 'cena_2' -> ['cena_', 2, ''].
    (A ordem alfabética punha cena_10 logo a seguir a cena_1.)
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", Path(path).name)]


def read_num_boxes(path):
    """Lê o num_boxes.json, ignorando as entradas que começam por '_' (instruções e exemplos)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: int(v) for k, v in data.items() if not k.startswith("_")}


def download_scenes(url, scenes_dir):
    """Copia as cenas de um repositório GitHub para a pasta das cenas (test/).

    1. Faz 'git clone' do repositório para uma pasta temporária (só a última versão: --depth 1);
       aceita também o link copiado do browser (ex.: .../tree/main/cenas_teste), que é cortado;
    2. Procura todas as pastas que são cenas (têm a subpasta point_clouds/), em qualquer sítio do repositório;
    3. Copia cada cena para test/<cena> (as que já lá estão não são copiadas outra vez);
    4. Apaga a pasta temporária.
    """
    if "/tree/" in url:                 # link de uma pasta no browser -> link do repositório
        url = url.split("/tree/")[0]
    scenes_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        print(f"A transferir as cenas de {url} (pode demorar um pouco)...")
        result = subprocess.run(["git", "clone", "--depth", "1", url, tmp], capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"Não foi possível transferir o repositório:\n{result.stderr}")
        copied = 0
        for point_clouds in sorted(Path(tmp).rglob("point_clouds"), key=lambda p: natural_key(p.parent)):
            scene = point_clouds.parent
            target = scenes_dir / scene.name
            if target.exists():
                print(f"  {scene.name}: já existe em {scenes_dir.name}/, não foi copiada")
                continue
            # Copia só os dados da cena (nunca o ground truth, se por acaso existir)
            shutil.copytree(scene, target, ignore=shutil.ignore_patterns("ground_truth.json", "config.json"))
            copied += 1
        print(f"  {copied} cenas copiadas para {scenes_dir}")


def git_push(output, message):
    """git add + commit + push. Um erro aqui é mostrado mas não interrompe o script."""
    commands = [["git", "add", str(output)],
                ["git", "commit", "-m", message],
                ["git", "push"]]
    for command in commands:
        result = subprocess.run(command, cwd=FOLDER, capture_output=True, text=True)
        if result.returncode != 0 and "nothing to commit" not in result.stdout:
            print(f"  [git] '{' '.join(command)}' falhou:\n{result.stdout}{result.stderr}")
            return False
    print("  [git] resultados enviados para o GitHub")
    return True


# #####################################################################################################
# JANELAS QUE NÃO PARAM O PROGRAMA
# #####################################################################################################
# As janelas 3D do Open3D só respondem ao rato enquanto o programa está a tratar delas. Para o programa
# continuar (e poder perguntar o n.º de caixas) com as janelas abertas, as janelas são abertas por um
# SEGUNDO processo: este programa grava os dados das janelas de cada parte (figure_data, só arrays) num
# .npz e um pedido .json; o segundo processo lê os pedidos, abre as janelas (make_windows da parte) e
# mantém-nas abertas e a responder ao rato.
def process_alive(pid):
    """True se o processo pid ainda está a correr."""
    if os.name == "nt":
        import ctypes                                # no Windows, os.kill(pid, 0) mataria o processo
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)       # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259                                  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class Windows:
    """Lado deste programa: lança o processo das janelas e envia-lhe pedidos."""

    def __init__(self, show=True):
        shutil.rmtree(WINDOWS_FOLDER, ignore_errors=True)       # pedidos de corridas anteriores
        self.folder = WINDOWS_FOLDER / time.strftime("%Y%m%d_%H%M%S")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.count = 0
        command = [sys.executable, str(Path(__file__).resolve()), "--janelas", str(self.folder),
                   "--pid", str(os.getpid())] + ([] if show else ["--sem_janelas"])
        # Prioridade baixa: as janelas não tiram tempo às partes que estão a correr
        flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
        self.log_path = self.folder / "janelas.log"
        self.log = open(self.log_path, "w", encoding="utf-8")
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=self.log,
                                        stderr=subprocess.STDOUT, creationflags=flags)

    def add(self, data, title, png):
        """Pede uma janela: data = figure_data de uma parte; png = onde a figura é gravada."""
        self.count += 1
        name = f"{self.count:03d}"
        Path(png).parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.folder / f"{name}.npz", **data)
        request = {"dados": f"{name}.npz", "parte": str(data["parte"]), "titulo": title, "png": str(png)}
        tmp = self.folder / f"{name}.tmp"
        tmp.write_text(json.dumps(request), encoding="utf-8")
        tmp.rename(self.folder / f"{name}.json")      # o .json só aparece quando está completo

    def finish(self, wait=False):
        """Diz ao processo das janelas que não há mais pedidos (e, com wait, espera que acabe)."""
        (self.folder / "fim").write_text("", encoding="utf-8")
        if wait:
            self.process.wait()
        self.log.close()


def collect(views, make_data, part):
    """Guarda os dados das vistas de uma parte (figure_data) para mostrar no fim da cena. Um erro aqui é
    mostrado mas não interrompe a cena."""
    try:
        views.append((make_data(), part))
    except Exception as error:
        print(f"  (vistas de {part} não criadas: {error})")


def run_scene(scene, output, views, show, known, push=False):
    """Corre uma cena do princípio ao fim: as 4 partes (savi/) e o avaliador.

    As vistas de cada parte são juntadas a views; show() abre a janela da cena quando as 4 partes
    acabam. Só depois o avaliador pede o número real de caixas (known: os do num_boxes.json)."""
    import avaliador
    import comum
    import extra_fotometria as extra
    import parte1_registo as parte1
    import parte2_segmentacao as parte2
    import parte3_ajuste as parte3

    result_path = output / scene.name / "result.json"
    # Parte 1: registo -> test/<cena>/partes/registo.json
    r1 = parte1.run_registration(scene, scene / comum.OUTPUT_FOLDER)
    collect(views, lambda: parte1.figure_data(r1), "parte1_registo")
    # Extra: fotometria -> test/<cena>/partes/fotometria.json (um erro aqui não impede o resto)
    try:
        rx = extra.run_photometry(r1)
        collect(views, lambda: extra.figure_data(r1, rx), "extra_fotometria")
    except Exception:
        traceback.print_exc()
        print(f"  ERRO na fotometria da cena {scene.name} (não impede o resto)")
    # Parte 2: segmentação -> test/<cena>/partes/objetos.ply, mesa.json, segmentacao.npz
    r2 = parte2.run_segmentation(r1)            # estima o número de caixas
    collect(views, lambda: parte2.figure_data(r1, r2), "parte2_segmentacao")
    # Parte 3: ajuste -> results_test/<cena>/result.json
    r3 = parte3.run_fit(r1, r2)
    parte3.write_scene_result(result_path, r1, r2, r3)
    print(f"  -> {result_path}")
    try:
        # Figura da vista de cima (se falhar, o result.json já está escrito)
        parte3.save_top_view(result_path.parent / "top_view.png", r2["seg"].points,
                             r2["seg"].colors, r3["params_opt"], r3["params_bbox"], title=scene.name)
    except Exception as error:
        print(f"  (figura não gravada: {error})")
    # Caixas verdadeiras (só se a cena tiver ground truth): só para desenhar e avaliar
    gt = comum.load_ground_truth(scene)
    gt_params = parte3.gt_params_in_table(gt, r2["table"].T_camera0_table) if gt else None
    collect(views, lambda: parte3.figure_data(r1, r2, r3, gt_params), "parte3_ajuste")
    if push:
        git_push(output, f"Resultado da cena {scene.name}")
    show()                                      # janela da cena (as 4 partes já acabaram)
    # Avaliador: SÓ AGORA é pedido o número real de caixas (as 4 partes não o usaram)
    real = known[scene.name] if scene.name in known else avaliador.ask_num_boxes(scene.name)
    avaliador.print_box_count(avaliador.found_boxes(r2), real)
    if gt is None:
        print("  avaliador: a cena não tem ground_truth.json, não há mais nada a comparar")
    else:
        try:
            avaliador.print_evaluation({**r1, **r2, **r3, "ground_truth": gt})
        except Exception:
            traceback.print_exc()
            print(f"  ERRO no avaliador da cena {scene.name}")


def open_windows(folder, show, parent_pid):
    """Lado do processo das janelas: abre as janelas de cada pedido que aparece na pasta e grava cada uma
    em PNG.

    Termina quando o programa principal acabou (ou foi interrompido) e todas as janelas foram fechadas.
    """
    import visualizacao                              # janelas 3D do Open3D (SceneWindows)
    folder = Path(folder)
    windows = visualizacao.SceneWindows(show)
    seen, finished = set(), False
    while True:
        for path in sorted(folder.glob("*.json")):
            if path.name in seen:
                continue
            seen.add(path.name)
            try:
                request = json.loads(path.read_text(encoding="utf-8"))
                with np.load(folder / request["dados"]) as npz:
                    data = {k: (npz[k].item() if npz[k].ndim == 0 else npz[k]) for k in npz.files}
                png = Path(request["png"])
                for spec in importlib.import_module(request["parte"]).make_windows(data):
                    windows.add(spec, png.with_name(png.stem + spec["name"] + ".png"), group=str(data["cena"]))
                print(f"{request['titulo']}: {png}", flush=True)
            except Exception:
                traceback.print_exc()
                sys.stdout.flush()
        if (folder / "fim").exists() or not process_alive(parent_pid):
            finished = True
        if finished and not windows.windows and not set(p.name for p in folder.glob("*.json")) - seen:
            break
        if windows.windows:
            windows.tick()                           # as janelas respondem ao rato e as animações avançam
            time.sleep(0.01)
        else:
            time.sleep(0.3)
    sys.stdout.flush()
    os._exit(0)          # sai já (o Open3D às vezes falha ao fechar e o erro não interessa)


# #####################################################################################################
# PROGRAMA PRINCIPAL
# #####################################################################################################
def main_test():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", default=str(FOLDER / "test"), help="pasta com as cenas")
    parser.add_argument("--output", default=str(FOLDER / "results_test"), help="pasta dos resultados")
    parser.add_argument("--git", action="store_true", help="commit + push depois de cada cena")
    parser.add_argument("--redo", action="store_true", help="volta a correr cenas que já têm resultado")
    parser.add_argument("--download", metavar="URL", help="copia primeiro as cenas deste repositório GitHub para --scenes")
    parser.add_argument("--sem_janelas", action="store_true", help="não abre janelas (só grava as figuras)")
    parser.add_argument("--janelas", help=argparse.SUPPRESS)       # uso interno: processo das janelas
    parser.add_argument("--pid", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.janelas:
        open_windows(args.janelas, not args.sem_janelas, args.pid)
        return

    scenes_dir, output = Path(args.scenes), Path(args.output)
    if args.download:
        download_scenes(args.download, scenes_dir)
    # Uma cena é qualquer subpasta com point_clouds/
    # Ordem natural (cena_1, cena_2, ..., cena_10) e não alfabética (cena_1, cena_10, cena_2, ...)
    scenes = sorted((p for p in scenes_dir.iterdir() if (p / "point_clouds").is_dir()), key=natural_key)
    if not scenes:
        print(f"Não há cenas em {scenes_dir} (cada cena tem de ter a subpasta point_clouds/)")
        return

    known = read_num_boxes(scenes_dir / "num_boxes.json")
    todo = [s for s in scenes if args.redo or not (output / s.name / "result.json").exists()]
    for s in scenes:
        if s not in todo:
            print(f"{s.name}: já tem resultado, saltada")
    if not todo:
        return
    start = time.time()
    failed, done = [], []
    windows = Windows(show=not args.sem_janelas)

    def show(scene, views):
        """Abre a janela da cena com as vistas juntadas até aqui (e esvazia a lista)."""
        for data, part in views:
            try:
                windows.add(data, f"{scene.name} - {part}", Path(output) / scene.name / f"{part}.png")
            except Exception as error:
                print(f"  (vistas de {part} não mostradas: {error})")
        views.clear()

    # Uma cena de cada vez: as 4 partes, a janela da cena com as vistas todas e o avaliador.
    for k, scene in enumerate(todo, 1):
        print(f"\n===== [{k}/{len(todo)}] {scene.name} - {(time.time() - start) / 60:.1f} min desde o início =====")
        views = []                         # vistas da cena: só são mostradas quando as 4 partes acabam
        try:
            run_scene(scene, output, views, lambda: show(scene, views), known, args.git)
            done.append(scene.name)
        except Exception:
            # Não deixa uma cena com problemas impedir as restantes
            traceback.print_exc()
            print(f"  ERRO na cena {scene.name}: passa à seguinte")
            failed.append(scene.name)
        finally:
            show(scene, views)             # com erro: mostra o que ficou feito

    print(f"\nTerminado em {(time.time() - start) / 60:.1f} min. "
          f"Feitas: {len(done)}  Com erro: {failed if failed else 'nenhuma'}")
    windows.finish(wait=args.sem_janelas)
    if args.sem_janelas:
        print(f"Figuras gravadas em {output}/<cena>/")
    else:
        print(f"As janelas continuam abertas até serem fechadas (figuras também em {output}/<cena>/; "
              f"erros das janelas em {windows.log_path})")


if __name__ == "__main__":
    main_test()
