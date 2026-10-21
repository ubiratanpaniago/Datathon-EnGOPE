# Datathon EnGOPE

Solução da nossa equipe para o **III Datathon IME-UFG**, parte do VII Encontro Goiano de Probabilidade e Estatística (ENGOPE). O parceiro do desafio é a SIAN, do Grupo HP.

## O desafio

Estimar o preço de um contrato de licitação de 3 anos, a partir de dados dos integrantes do Grupo HP. São três trilhas:

1. **Rotatividade:** projetar o turnover por empresa.
2. **Folguistas e feristas:** dimensionar o quadro extra para cobrir ausências.
3. **Escala 6x1:** dimensionar o quadro extra se a escala mudar para 5x2.

## Estrutura do projeto

```
├── dados/        # dados fornecidos (CSVs)
├── docs/
│   ├── enunciado/  # instruções do desafio
│   ├── regras/     # regulamento, critérios de avaliação e resumo das regras
│   └── problema/   # explicação do problema e das trilhas
├── src/          # código da solução
├── notebooks/    # exploração e análises
└── relatorio/    # relatório final em PDF
```

## Documentação

- [Regras, datas e formato de envio](docs/regras/REGRAS.md)
- [O problema e as trilhas](docs/problema/PROBLEMA.md)

## Entrega

Prazo de submissão: **21/10 às 19h00**. Enviar por e-mail um PDF de até 10 páginas e o código da solução.

## Como executar

**Forma mais simples (arquivo único, enviado na submissão):** `pip install pandas numpy scipy statsmodels` e `python solucao_datathon.py`, com os CSVs em `dados/`. O script reproduz todos os resultados de `resultados/` em cerca de 2 minutos (inclui a auditoria dos dados e as extensões do notebook 06). Com `matplotlib` instalado, também salva os 8 gráficos em `figuras/` (sem ele, os gráficos são pulados e os resultados não mudam).

**Notebooks (com gráficos e texto):**

1. `pip install -r requirements.txt`
2. Colocar os CSVs fornecidos em `dados/` (a pasta não vai no pacote de entrega, porque a organização já tem os dados).
3. Rodar os notebooks de `notebooks/` em ordem: `01` perfil dos dados, `02` Trilha 1, `03` Trilha 2, `04` Trilha 3, `05` modelo integrado. Cada um lê `../dados` e os resultados anteriores em `../resultados`, e grava os seus em `../resultados`. A ordem importa porque o `05` lê os resultados dos `02` a `04`.

O código reutilizável está em `src/` (`rotatividade.py`, `ausencias.py`, `escala.py`, `integrado.py`). O resumo do que foi feito está em [`docs/RESUMO_PREVIO.md`](docs/RESUMO_PREVIO.md) e o relatório em `relatorio/Relatorio_Datathon.pdf`.
