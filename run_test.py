#!/usr/bin/env python3
"""Dia da apresentação: corre o sistema em TODAS as cenas de uma pasta, uma de cada vez.

Utilização:
    python run_test.py --download <link do repositório GitHub>   # copia as cenas do GitHub para test/ e corre
    python run_test.py              # corre todas as cenas que já estão em test/
    python run_test.py --git        # e faz git add + commit + push depois de cada cena
    python run_test.py --redo       # volta a correr cenas que já têm resultado

Para cada cena test/<cena>/ escreve results_test/<cena>/result.json (e a figura top_view.png).
O número de caixas de cada cena vem de test/num_boxes.json; se faltar, é perguntado no terminal.

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
import trabalho1_savi as t1                       # noqa: E402 (tem de vir depois do sys.path)


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

    # Primeiro recolhe o número de caixas de TODAS as cenas, para depois correr sem interrupções
    known = read_num_boxes(scenes_dir / "num_boxes.json")
    num_boxes = {s.name: known[s.name] if s.name in known else ask_num_boxes(s.name) for s in scenes}
    print("\nCenas a processar:", ", ".join(f"{k} ({v} caixas)" for k, v in num_boxes.items()))

    start = time.time()
    done, failed = [], []
    for k, scene in enumerate(scenes, 1):
        result_path = output / scene.name / "result.json"
        if result_path.exists() and not args.redo:
            print(f"\n[{k}/{len(scenes)}] {scene.name}: já tem resultado, saltada")
            continue

        print(f"\n[{k}/{len(scenes)}] {scene.name} ({num_boxes[scene.name]} caixas)"
              f" - {(time.time() - start) / 60:.1f} min desde o início")
        try:
            r = t1.run(scene, num_boxes[scene.name])
            boxes = t1.boxes_to_camera0(r["params_opt"], r["table"].T_camera0_table)
            t1.write_result(result_path, [v.name for v in r["scene"].views], r["T_cams"], boxes)
            print(f"  -> {result_path}")
            try:
                # Figura da vista de cima (se falhar, o result.json já está escrito)
                t1.save_top_view(result_path.parent / "top_view.png", r["seg"].points,
                                 r["seg"].colors, r["params_opt"], r["params_bbox"], title=scene.name)
            except Exception as error:
                print(f"  (figura não gravada: {error})")
            done.append(scene.name)
            if args.git:
                git_push(output, f"Resultado da cena {scene.name}")
        except Exception:
            # Não deixa uma cena com problemas impedir as restantes
            traceback.print_exc()
            print(f"  ERRO na cena {scene.name}: passa à seguinte")
            failed.append(scene.name)

    print(f"\nTerminado em {(time.time() - start) / 60:.1f} min. "
          f"Feitas: {len(done)}  Com erro: {failed if failed else 'nenhuma'}")


if __name__ == "__main__":
    main_test()
