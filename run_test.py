#!/usr/bin/env python3
"""Dia da apresentação: corre o sistema em TODAS as cenas de uma pasta, uma de cada vez.

Utilização:
    python run_test.py --download <link do repositório GitHub>   # copia as cenas do GitHub para test/ e corre
    python run_test.py              # corre todas as cenas que já estão em test/
    python run_test.py --git        # e faz git add + commit + push depois de cada cena
    python run_test.py --redo       # volta a correr cenas que já têm resultado
    python run_test.py --sem_janelas   # não abre janelas (só grava as figuras em results_test/<cena>/)

Corre os 4 ficheiros (savi/) um depois do outro e, no fim de cada cena, o avaliador, em duas fases:
    FASE 1. em TODAS as cenas (não precisa do número de caixas, corre sem interrupções):
            parte1_registo.py      -> test/<cena>/partes/registo.json   (pose de cada câmara)
            extra_fotometria.py    -> test/<cena>/partes/fotometria.json (exposição de cada imagem)
    FASE 2. só depois de a fase 1 acabar, cena a cena. É aqui que é pedido o número de caixas:
            parte2_segmentacao.py  -> test/<cena>/partes/objetos.ply, mesa.json, segmentacao.npz
            parte3_ajuste.py       -> results_test/<cena>/result.json (e top_view.png)
            avaliador.py           -> erros em relação ao ground truth (só se a cena o tiver)
O número de caixas de uma cena que esteja em test/num_boxes.json não é perguntado.

Janelas: cada parte abre a sua janela (com subplots) logo que acaba, sem parar o programa: as janelas
são desenhadas por um segundo processo (python run_test.py --janelas ...), por isso pode ver-se, por
exemplo, a cena 3 sem fechar as janelas das cenas 1 e 2, e o programa continua enquanto estão abertas.
As figuras são gravadas também em results_test/<cena>/ (parte1_registo.png, extra_fotometria.png,
parte2_segmentacao.png, parte3_ajuste.png, parte3_ajuste_otimizacao.png).

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


def ask_num_boxes(name):
    """Pergunta no terminal o número de caixas de uma cena que não está no num_boxes.json."""
    while True:
        answer = input(f"Número de caixas da cena '{name}': ").strip()
        if answer.isdigit() and int(answer) > 0:
            return int(answer)
        print("  escreve um número inteiro positivo")


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
# O matplotlib só desenha janelas enquanto o programa está parado à espera delas (plt.show). Para o
# programa continuar (e poder perguntar o n.º de caixas) com as janelas abertas, as janelas são
# desenhadas por um SEGUNDO processo: este programa grava os dados de cada janela (figure_data de cada
# parte, só arrays) num .npz e um pedido .json; o segundo processo lê os pedidos, desenha as janelas
# (make_figures da parte) e mantém-nas abertas e a responder ao rato.
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


def add_window(windows, make_data, scene, part, output):
    """Pede a janela de uma parte de uma cena. Um erro aqui é mostrado mas não interrompe o script."""
    try:
        windows.add(make_data(), f"{scene.name} - {part}", Path(output) / scene.name / f"{part}.png")
    except Exception as error:
        print(f"  (janela de {part} não criada: {error})")


def open_windows(folder, show, parent_pid):
    """Lado do processo das janelas: desenha cada pedido que aparece na pasta e grava a figura em PNG.

    Termina quando o programa principal acabou (ou foi interrompido) e todas as janelas foram fechadas.
    """
    folder = Path(folder)
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.raise_window"] = False     # uma janela nova não salta para a frente das outras
    seen, animations, finished = set(), [], False
    while True:
        for path in sorted(folder.glob("*.json")):
            if path.name in seen:
                continue
            seen.add(path.name)
            try:
                request = json.loads(path.read_text(encoding="utf-8"))
                with np.load(folder / request["dados"]) as npz:
                    data = {k: (npz[k].item() if npz[k].ndim == 0 else npz[k]) for k in npz.files}
                figures = importlib.import_module(request["parte"]).make_figures(data)
                png = Path(request["png"])
                for f in figures:
                    f["fig"].savefig(png.with_name(png.stem + f["name"] + ".png"), dpi=110)
                    if show:
                        f["fig"].canvas.manager.set_window_title(request["titulo"] + f["name"])
                        f["fig"].show()
                        if f.get("start"):
                            animations.append(f["start"]())     # guardadas, senão param
                    else:
                        plt.close(f["fig"])
                print(f"{request['titulo']}: {png}", flush=True)
            except Exception:
                traceback.print_exc()
                sys.stdout.flush()
        if (folder / "fim").exists() or not process_alive(parent_pid):
            finished = True
        figures_open = plt.get_fignums()
        if finished and (not show or not figures_open) and not set(p.name for p in folder.glob("*.json")) - seen:
            break
        if show and figures_open:
            # Deixa as janelas responder ao rato (e as animações correr) durante 0.3 s.
            # (plt.pause traria todas as janelas para a frente de cada vez)
            plt.figure(figures_open[0]).canvas.start_event_loop(0.3)
        else:
            time.sleep(0.3)


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

    # As 4 partes e o avaliador (só agora: o processo das janelas não precisa deles)
    import avaliador
    import extra_fotometria as extra
    import parte1_registo as parte1
    import parte2_segmentacao as parte2
    import parte3_ajuste as parte3

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
    failed = []
    windows = Windows(show=not args.sem_janelas)

    # ---------------- FASE 1: Parte 1 (registo) e extra (fotometria) em todas as cenas ----------------
    # Não precisam do número de caixas, por isso correm sem interrupções.
    print(f"\n===== FASE 1: registo (parte1_registo.py) e fotometria (extra_fotometria.py) em {len(todo)} cenas =====")
    registered = []
    for k, scene in enumerate(todo, 1):
        print(f"\n[{k}/{len(todo)}] {scene.name} - {(time.time() - start) / 60:.1f} min desde o início")
        try:
            # Pose de cada câmara em test/<cena>/partes/registo.json
            r1 = parte1.run_registration(scene, scene / parte1.OUTPUT_FOLDER)
            add_window(windows, lambda: parte1.figure_data(r1), scene, "parte1_registo", output)
            registered.append(scene)
        except Exception:
            # Não deixa uma cena com problemas impedir as restantes
            traceback.print_exc()
            print(f"  ERRO no registo da cena {scene.name}: passa à seguinte")
            failed.append(scene.name)
            continue
        try:
            rx = extra.run_photometry(r1)
            add_window(windows, lambda: extra.figure_data(r1, rx), scene, "extra_fotometria", output)
        except Exception:
            traceback.print_exc()
            print(f"  ERRO na fotometria da cena {scene.name} (não impede o resto)")
        del r1                                  # liberta a nuvem agregada antes da cena seguinte

    # ---------------- FASE 2: Partes 2 e 3 e avaliador, cena a cena ----------------
    # Só agora, com a fase 1 feita em todas as cenas, é pedido o número de caixas de cada cena.
    print(f"\n===== FASE 2: segmentação (parte2_segmentacao.py), ajuste (parte3_ajuste.py) "
          f"e avaliador em {len(registered)} cenas =====")
    done = []
    for k, scene in enumerate(registered, 1):
        result_path = output / scene.name / "result.json"
        print(f"\n[{k}/{len(registered)}] {scene.name}")
        n = known[scene.name] if scene.name in known else ask_num_boxes(scene.name)
        try:
            r1 = parte1.load_registration(scene / parte1.OUTPUT_FOLDER)
            r2 = parte2.run_segmentation(r1, n)
            add_window(windows, lambda: parte2.figure_data(r2), scene, "parte2_segmentacao", output)
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
            gt = parte1.load_ground_truth(scene)
            gt_params = parte3.gt_params_in_table(gt, r2["table"].T_camera0_table) if gt else None
            add_window(windows, lambda: parte3.figure_data(r2, r3, gt_params), scene, "parte3_ajuste", output)
            done.append(scene.name)
            if args.git:
                git_push(output, f"Resultado da cena {scene.name}")
        except Exception:
            traceback.print_exc()
            print(f"  ERRO na segmentação / ajuste da cena {scene.name}: passa à seguinte")
            failed.append(scene.name)
            continue
        # Avaliador: erros em relação ao ground truth
        if gt is None:
            print("  avaliador: a cena não tem ground_truth.json, não há nada a comparar")
        else:
            try:
                avaliador.print_evaluation({**r1, **r2, **r3, "ground_truth": gt})
            except Exception:
                traceback.print_exc()
                print(f"  ERRO no avaliador da cena {scene.name}")

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
