# Teste — Dia da apresentação (simulação n.º 4: 15 cenas de todos os tipos)

As cenas estão na pasta `cenas_teste/`. Foram geradas pelo simulador e **não têm** `ground_truth.json` nem `config.json`.

Cada cena tem **no máximo 4 imagens** (2, 3 ou 4). As câmaras estão num arco, com 45° a 75° entre câmaras vizinhas,
e a distância e a altura de cada câmara variam. Há faces das caixas que nenhuma câmara vê.

| cena | caixas | imagens |
|---|---|---|
| cena_1 | 1 | 4 |
| cena_2 | 1 | 2 |
| cena_3 | 2 | 3 |
| cena_4 | 3 | 4 |
| cena_5 | 2 | 4 |
| cena_6 | 2 | 3 |
| cena_7 | 2 | 4 |
| cena_8 | 2 | 4 |
| cena_9 | 3 | 4 |
| cena_10 | 2 | 4 |
| cena_11 | 3 | 4 |
| cena_12 | 3 | 3 |
| cena_13 | 2 | 4 |
| cena_14 | 2 | 2 |
| cena_15 | 4 | 4 |

## O que tem de fazer

1. Obter as cenas deste repositório e pô-las onde o seu programa as consegue ler;
2. Correr o seu programa em cada cena;
3. Fazer *commit* e *push*, para este repositório, de um `result.json` por cena,
   em `results_test/<cena>/result.json` (por exemplo `results_test/cena_1/result.json`).

## Avaliação

O desempenho é o IoU 3D médio entre cada caixa estimada e a caixa verdadeira correspondente.
Uma caixa não encontrada, ou uma cena sem `result.json`, conta com IoU = 0.
