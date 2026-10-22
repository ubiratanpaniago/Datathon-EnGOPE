<div align="center">

# Quanto custa manter o serviço por três anos?

**Rotatividade, cobertura de ausências e mudança de escala no Grupo HP**

III Datathon IME-UFG · VII ENGOPE · parceiro **SIAN (Grupo HP)**
Equipe **Anovas**

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)
![Relatório](https://img.shields.io/badge/relat%C3%B3rio-PDF-B31B1B)

[**Relatório final (PDF)**](relatorio/Datathon_Anovas_Relatorio_Final.pdf) · [Fonte LaTeX](relatorio/main.tex) · [Auditoria dos dados (PDF)](auditoria/relatorio_problemas_dados.pdf)

</div>

---

## O desafio

Para disputar uma licitação de três anos, o Grupo HP precisa saber **quantas pessoas manter na folha** para entregar o serviço todos os dias previstos. O desafio tem três trilhas, e cada uma estima uma parte desse custo:

| Trilha | Pergunta |
|---|---|
| **T1 · Rotatividade** | Qual será o turnover de cada empresa ao longo do contrato? |
| **T2 · Folguistas e feristas** | Quantos integrantes a mais são necessários para cobrir férias, atestados e afastamentos? |
| **T3 · Escala 6x1 → 5x2** | Quantos integrantes a mais seriam necessários se a escala mudar? |

Tratamos as três trilhas como partes de **um único modelo de dimensionamento**, cuja unidade é a *posição* (a pessoa escalada num dia). Isso evita contar a mesma cobertura duas vezes:

```
folha / posição = (1 + s·(f − 1)) / (1 − a) × (1 + T · dias_vaga / 365)
                   └── escala (T3) ──┘ └ ausências (T2) ┘ └── vagas (T1) ──┘
```

## Principais resultados

| | Resultado |
|---|---|
| **Folha por 100 posições** | de **119 a 153 pessoas**, conforme a empresa |
| **Operação atual** | a folha de **4.727** pessoas vai a **4.967** (+240, +5,1%) |
| **Acréscimo de custo** | cerca de **R$ 30 milhões em três anos** (encargos de 1,7×, premissa) |
| **Maior incerteza** | **76%** do quadro operacional não tem ponto nem afastamentos |

<details>
<summary><b>Resultados por trilha</b></summary>

**T1 · Rotatividade** (média de 36 meses, escolhida por backtest; faixa P10–P90 validada fora da amostra)

| Empresa | Turnover projetado (% a.a.) | Faixa | Confiança |
|---|---|---|---|
| 5 | 10,2 | 7,6 a 12,8 | alta |
| 8 | 28,5 | 21,7 a 34,5 | média-alta |
| 6 | 61,1 | 35,8 a 68,9 | baixa |

**T2 · Folguistas e feristas** (quadro extra, % das posições, ano 1)

| Empresa 5 | Empresa 8 | Empresa 6 |
|---|---|---|
| 21,6% | 20,1% | 18,0% |

Com a reserva escolhida por custo esperado (em vez da regra de 95%), o quadro extra cai de 4 a 6 p.p. se o posto descoberto puder ser coberto com hora extra.

**T3 · Escala 6x1 → 5x2**

O ponto mostra que o 6x1 trabalha **42,7 h/semana** e o 5x2 atual **43,0 h/semana**: a mesma carga em menos dias. O adicional depende da jornada após a mudança:

| Cenário | Fator de folha | 6x1 no cadastro (736) | Incluindo escala desconhecida (2.653) |
|---|---|---|---|
| Jornada diária mantida | 1,197 | **+145** | +522 |
| Limite semanal de 40 h | 1,10 | +74 | +265 |
| Carga semanal mantida | 1,00 | 0 | 0 |

</details>

## Estrutura do repositório

```
.
├── relatorio/      Relatório final em PDF e a fonte LaTeX
├── auditoria/      Script e PDF da auditoria dos dados (20 problemas, com testes estatísticos)
├── notebooks/      A análise completa, em ordem (01 a 06)
├── src/            Funções usadas pelos notebooks
├── resultados/     Tabelas finais (CSV) geradas pelos notebooks
├── dados/          Os 14 CSVs fornecidos pela organização
└── docs/           Enunciado, regras do evento e documentos de trabalho
```

| Notebook | Conteúdo |
|---|---|
| `01_perfil_dos_dados` | Auditoria: estrutura, integridade, cobertura do ponto e dos afastamentos |
| `02_trilha1_rotatividade` | T1: série mensal, diagnóstico, backtest de 8 estratégias e coortes, projeção por empresa |
| `03_trilha2_folguistas_feristas` | T2: limpeza do ponto, perda de carga em três blocos, reserva, credibilidade e validação |
| `04_trilha3_escala` | T3: jornada medida no ponto, cenários de 6x1 para 5x2, sensibilidade da rotatividade |
| `05_modelo_integrado` | Modelo único por posição e custo em R$ |
| `06_extensoes` | Faixas validadas fora da amostra, bayes empírico, reserva por custo esperado, incerteza das células sem dado e escala × rotatividade |

| Módulo (`src/`) | Função |
|---|---|
| `rotatividade.py` | Série mensal, turnover, risco por tempo de casa e simulação de coortes |
| `rotatividade_eb.py` | Encolhimento bayesiano empírico e faixas calibradas fora da amostra |
| `ausencias.py` | Classificação e limpeza do ponto, taxas por causa e quadro extra |
| `reserva_custo.py` | Reserva de menor custo esperado (a reserva também falta) |
| `escala.py` | Jornada praticada por escala e cenários de mudança |
| `integrado.py` | Modelo integrado por posição e custos |
| `efeito_escala_rotatividade.py` | Modelo de Poisson: escala praticada × taxa de desligamento |

## Como executar

```bash
# 1. Dependências (Python 3.11 ou superior)
pip install -r requirements.txt

# 2. Dados: os 14 CSVs da organização em dados/ (já incluídos neste repositório)

# 3. Executar os notebooks em ordem, a partir da pasta notebooks/
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

- Cada notebook grava suas tabelas em `resultados/`, e os gráficos ficam no próprio notebook.
- **A ordem importa:** do `04` em diante, cada notebook lê tabelas gravadas pelos anteriores (o `04` e o `06` usam as do `02` e do `03`; o `05`, as do `02` ao `04`).
- A execução completa leva cerca de 3 minutos e reproduz as 18 tabelas de `resultados/`.

**Auditoria dos dados.** O PDF já está em `auditoria/`. Para gerá-lo de novo:

```bash
pip install reportlab
python auditoria/relatorio_problemas_dados.py dados auditoria/relatorio_problemas_dados.pdf figuras
```

O script usa fontes do Windows. Em outro sistema, ajuste o caminho das fontes no início do arquivo.

## Premissas

As escolhas que não vêm dos dados estão declaradas no código e podem ser alteradas:

| Premissa | Valor |
|---|---|
| Encargos sobre o salário | 1,7× |
| Prazo para repor uma saída | 30 dias |
| Nível de serviço da reserva | 95% (ou custo *k* do posto descoberto) |
| Peso de credibilidade das células | *n* / (*n* + 100) |
| Data de corte dos dados | 30/09/2026 |

## Equipe Anovas

Augusto Duarte Farias · Ubiratan Alves Paniago Filho · Gustavo Bernardo Britto · Lygia Machado Rebouças

## Documentação

- [Regras, datas e formato de envio](docs/regras/REGRAS.md)
- [O problema e as trilhas](docs/problema/PROBLEMA.md)
- [Enunciado original](docs/enunciado/Instrucoes_Datathon_UFG.pdf)
