# Teste — Dia da apresentação (simulação n.º 7: 10 cenas difíceis)

As cenas estão na pasta `cenas_teste/`. Foram geradas pelo simulador e **não têm** `ground_truth.json` nem `config.json`.

Cada cena tem entre **2 e 4 imagens** (câmaras num arco, 45° a 75° entre vizinhas). Muitas cenas têm caixas da
**mesma cor** encostadas ou empilhadas.

| cena | caixas | imagens |
|---|---|---|
| cena_1 | 2 | 3 |
| cena_2 | 3 | 4 |
| cena_3 | 2 | 3 |
| cena_4 | 3 | 4 |
| cena_5 | 4 | 4 |
| cena_6 | 3 | 3 |
| cena_7 | 3 | 2 |
| cena_8 | 4 | 4 |
| cena_9 | 2 | 2 |
| cena_10 | 4 | 4 |

## O que tem de fazer

1. Obter as cenas deste repositório e pô-las onde o seu programa as consegue ler;
2. Correr o seu programa em cada cena;
3. Fazer *commit* e *push*, para este repositório, do código, do Read_me e de um `result.json` por cena,
   em `results_test/<cena>/result.json`.

## Avaliação

O desempenho é o IoU 3D médio entre cada caixa estimada e a caixa verdadeira correspondente.
Uma caixa não encontrada, ou uma cena sem `result.json`, conta com IoU = 0.
