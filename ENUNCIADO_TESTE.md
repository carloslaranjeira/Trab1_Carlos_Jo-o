# Teste â€” Dia da apresentaÃ§Ã£o (simulaÃ§Ã£o n.Âº 9: 10 cenas aleatÃ³rias)

As cenas estÃ£o na pasta `cenas_teste/`. Foram geradas pelo simulador e **nÃ£o tÃªm** `ground_truth.json` nem `config.json`.

As cenas tÃªm dificuldades diferentes e entre **1 e 8 imagens**. Com 1 a 4 imagens, as cÃ¢maras estÃ£o num arco (45Â° a 75Â°
entre vizinhas); com 5 a 8, dÃ£o a volta Ã  mesa.

| cena | caixas | imagens |
|---|---|---|
| cena_1 | 2 | 1 |
| cena_2 | 2 | 8 |
| cena_3 | 1 | 4 |
| cena_4 | 3 | 3 |
| cena_5 | 2 | 6 |
| cena_6 | 4 | 6 |
| cena_7 | 4 | 2 |
| cena_8 | 2 | 5 |
| cena_9 | 3 | 7 |
| cena_10 | 3 | 3 |

## O que tem de fazer

1. Obter as cenas deste repositÃ³rio e pÃ´-las onde o seu programa as consegue ler;
2. Correr o seu programa em cada cena;
3. Fazer *commit* e *push*, para este repositÃ³rio, do cÃ³digo, do Read_me e de um `result.json` por cena,
   em `results_test/<cena>/result.json`.

## AvaliaÃ§Ã£o

O desempenho Ã© o IoU 3D mÃ©dio entre cada caixa estimada e a caixa verdadeira correspondente.
Uma caixa nÃ£o encontrada, ou uma cena sem `result.json`, conta com IoU = 0.
