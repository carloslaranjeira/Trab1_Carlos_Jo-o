#!/usr/bin/env python3
"""Dia da apresentação: corre o sistema em TODAS as cenas de uma pasta, uma de cada vez.

Utilização:
    python run_test.py --download <link do repositório GitHub>   # copia as cenas do GitHub para test/ e corre
    python run_test.py              # corre todas as cenas que já estão em test/
    python run_test.py --git        # e faz git add + commit + push depois de cada cena
    python run_test.py --redo       # volta a correr cenas que já têm resultado

Corre os 2 scripts um depois do outro, em duas fases:
    FASE 1. savi/detecao_caixas.py em TODAS as cenas -> test/<cena>/detecao/
            (nuvem de pontos só das caixas + câmaras e mesa; não precisa do número de caixas)
    FASE 2. só depois de a fase 1 acabar: savi/avaliador_caixas.py em cada cena. É aqui que é
            pedido, cena a cena, o número de caixas -> results_test/<cena>/result.json (e top_view.png)
O número de caixas de uma cena que esteja em test/num_boxes.json não é perguntado.

Pensado para não perder resultados:
    - um erro numa cena não pára as outras (fica registado e passa-se à seguinte);
    - cenas que já têm result.json são saltadas, por isso pode voltar-se a correr o script;
    - com --git, cada resultado é enviado para o GitHub logo que fica pronto.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

FOLDER = Path(__file__).resolve().parent         # pasta principal do trabalho
# O código está na pasta savi/: junta-a aos sítios onde o Python procura módulos
sys.path.insert(0, str(FOLDER / "savi"))
import avaliador_caixas as avaliador              # noqa: E402 (script 2; tem de vir depois do sys.path)
import detecao_caixas as detecao                  # noqa: E402 (script 1)


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


def main_test():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", default=str(FOLDER / "test"), help="pasta com as cenas")
    parser.add_argument("--output", default=str(FOLDER / "results_test"), help="pasta dos resultados")
    parser.add_argument("--git", action="store_true", help="commit + push depois de cada cena")
    parser.add_argument("--redo", action="store_true", help="volta a correr cenas que já têm resultado")
    parser.add_argument("--download", metavar="URL", help="copia primeiro as cenas deste repositório GitHub para --scenes")
    args = parser.parse_args()

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
    start = time.time()
    failed = []

    # ---------------- FASE 1: script 1 (deteção) em todas as cenas ----------------
    # Não precisa do número de caixas, por isso corre sem interrupções.
    print(f"\n===== FASE 1: deteção (script 1) em {len(todo)} cenas =====")
    detected = []
    for k, scene in enumerate(todo, 1):
        print(f"\n[{k}/{len(todo)}] {scene.name} - {(time.time() - start) / 60:.1f} min desde o início")
        try:
            # Nuvem de pontos só das caixas em test/<cena>/detecao/
            detecao.run_detection(scene, scene / "detecao")
            detected.append(scene)
        except Exception:
            # Não deixa uma cena com problemas impedir as restantes
            traceback.print_exc()
            print(f"  ERRO na deteção da cena {scene.name}: passa à seguinte")
            failed.append(scene.name)

    # ---------------- FASE 2: script 2 (avaliador) ----------------
    # Só agora, com a deteção de todas as cenas feita, é pedido o número de caixas de cada cena.
    print(f"\n===== FASE 2: avaliador (script 2) em {len(detected)} cenas =====")
    done = []
    for k, scene in enumerate(detected, 1):
        result_path = output / scene.name / "result.json"
        print(f"\n[{k}/{len(detected)}] {scene.name}")
        n = known[scene.name] if scene.name in known else ask_num_boxes(scene.name)
        try:
            r = avaliador.run(scene / "detecao", n)
            boxes = avaliador.boxes_to_camera0(r["params_opt"], r["table"].T_camera0_table)
            avaliador.write_result(result_path, r["images"], r["T_cams"], boxes)
            print(f"  -> {result_path}")
            try:
                # Figura da vista de cima (se falhar, o result.json já está escrito)
                avaliador.save_top_view(result_path.parent / "top_view.png", r["seg"].points,
                                        r["seg"].colors, r["params_opt"], r["params_bbox"], title=scene.name)
            except Exception as error:
                print(f"  (figura não gravada: {error})")
            done.append(scene.name)
            if args.git:
                git_push(output, f"Resultado da cena {scene.name}")
        except Exception:
            traceback.print_exc()
            print(f"  ERRO no avaliador da cena {scene.name}: passa à seguinte")
            failed.append(scene.name)

    print(f"\nTerminado em {(time.time() - start) / 60:.1f} min. "
          f"Feitas: {len(done)}  Com erro: {failed if failed else 'nenhuma'}")


if __name__ == "__main__":
    main_test()
