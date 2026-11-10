# Teste — Dia da apresentação (simulação n.º 6: 30 cenas)

As cenas estão na pasta `cenas_teste/`. Foram geradas pelo simulador e **não têm** `ground_truth.json` nem `config.json`.

Cada cena tem entre **1 e 4 imagens**. Nas cenas com uma só imagem não há registo a fazer, e há faces das caixas que não
se veem. Nas outras, as câmaras estão num arco, com 45° a 75° entre câmaras vizinhas.

| cena | caixas | imagens |
|---|---|---|
| cena_1 | 1 | 1 |
| cena_2 | 2 | 1 |
| cena_3 | 2 | 1 |
| cena_4 | 1 | 3 |
| cena_5 | 2 | 2 |
| cena_6 | 3 | 4 |
| cena_7 | 2 | 2 |
| cena_8 | 2 | 3 |
| cena_9 | 4 | 4 |
| cena_10 | 2 | 3 |
| cena_11 | 3 | 4 |
| cena_12 | 4 | 3 |
| cena_13 | 2 | 4 |
| cena_14 | 4 | 4 |
| cena_15 | 2 | 2 |
| cena_16 | 3 | 3 |
| cena_17 | 2 | 1 |
| cena_18 | 1 | 2 |
| cena_19 | 3 | 2 |
| cena_20 | 2 | 3 |
| cena_21 | 5 | 4 |
| cena_22 | 2 | 3 |
| cena_23 | 2 | 4 |
| cena_24 | 4 | 4 |
| cena_25 | 2 | 2 |
| cena_26 | 2 | 1 |
| cena_27 | 2 | 3 |
| cena_28 | 3 | 4 |
| cena_29 | 3 | 3 |
| cena_30 | 4 | 4 |

## O que tem de fazer

1. Obter as cenas deste repositório e pô-las onde o seu programa as consegue ler;
2. Correr o seu programa em cada cena;
3. Fazer *commit* e *push*, para este repositório, do código, do Read_me e de um `result.json` por cena,
   em `results_test/<cena>/result.json` (por exemplo `results_test/cena_1/result.json`).

## Avaliação

O desempenho é o IoU 3D médio entre cada caixa estimada e a caixa verdadeira correspondente.
Uma caixa não encontrada, ou uma cena sem `result.json`, conta com IoU = 0.
