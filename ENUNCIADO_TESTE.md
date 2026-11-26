# Teste — Dia da apresentação (simulação n.º 21: 5 cenas aleatórias, 2 com Exposure changes)

As cenas estão na pasta `cenas_teste/`. Foram geradas pelo simulador e **têm** `ground_truth.json` (para o avaliador comparar os resultados); não têm `config.json`.

As cenas têm dificuldades diferentes e entre **1 e 4 imagens**. As câmaras estão num arco (45° a 75° entre vizinhas).
**Duas cenas foram geradas com Exposure changes**: cada imagem tem o seu ganho s e o seu desvio b (o campo `photometry` do `ground_truth.json` tem os valores verdadeiros).

| cena | caixas | imagens | Exposure changes |
|---|---|---|---|
| cena_1 | 1 | 1 | não |
| cena_2 | 2 | 4 | não |
| cena_3 | 3 | 2 | sim |
| cena_4 | 3 | 4 | sim |
| cena_5 | 2 | 3 | não |

## O que tem de fazer

1. Obter as cenas deste repositório e pô-las onde o seu programa as consegue ler;
2. Correr o seu programa em cada cena;
3. Fazer *commit* e *push*, para este repositório, do código, do Read_me e de um `result.json` por cena,
   em `results_test/<cena>/result.json`.

## Avaliação

O desempenho é o IoU 3D médio entre cada caixa estimada e a caixa verdadeira correspondente.
Uma caixa não encontrada, ou uma cena sem `result.json`, conta com IoU = 0.
