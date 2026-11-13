# Teste — Dia da apresentação (simulação n.º 10: 10 cenas aleatórias)

As cenas estão na pasta `cenas_teste/`. Foram geradas pelo simulador e **não têm** `ground_truth.json` nem `config.json`.

As cenas têm dificuldades diferentes e entre **1 e 8 imagens**. Com 1 a 4 imagens, as câmaras estão num arco (45° a 75°
entre vizinhas); com 5 a 8, dão a volta à mesa.

| cena | caixas | imagens |
|---|---|---|
| cena_1 | 2 | 3 |
| cena_2 | 3 | 3 |
| cena_3 | 3 | 1 |
| cena_4 | 2 | 6 |
| cena_5 | 1 | 8 |
| cena_6 | 2 | 5 |
| cena_7 | 4 | 7 |
| cena_8 | 4 | 6 |
| cena_9 | 3 | 4 |
| cena_10 | 2 | 2 |

## O que tem de fazer

1. Obter as cenas deste repositório e pô-las onde o seu programa as consegue ler;
2. Correr o seu programa em cada cena;
3. Fazer *commit* e *push*, para este repositório, do código, do Read_me e de um `result.json` por cena,
   em `results_test/<cena>/result.json`.

## Avaliação

O desempenho é o IoU 3D médio entre cada caixa estimada e a caixa verdadeira correspondente.
Uma caixa não encontrada, ou uma cena sem `result.json`, conta com IoU = 0.
