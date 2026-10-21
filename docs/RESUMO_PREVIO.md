# Resumo prévio: o que conseguimos e como chegamos lá

Documento de trabalho, **não é o relatório final** (o PDF de até 10 páginas ainda será escrito). Os números vêm dos notebooks em `notebooks/` e dos CSVs em `resultados/`.

## 1. Visão geral

O desafio é estimar o preço de um contrato de 3 anos do Grupo HP. Cada trilha estima uma parte do custo, e no fim elas foram reunidas num único modelo:

| Trilha | Pergunta | Resposta em uma linha |
|---|---|---|
| **T1** Rotatividade | quanto de turnover por empresa nos 3 anos? | de ~10% (empresa 5) a ~61% (empresa 6), com faixa de incerteza por empresa |
| **T2** Folguistas e feristas | quantos a mais para cobrir ausências? | de ~16% a ~22% das posições nas empresas grandes |
| **T3** Escala 6x1 para 5x2 | quantos a mais se a escala mudar? | de **0 a +20%** da folha em 6x1, conforme a jornada; **145 pessoas** no caso base |
| **Integrado** | quanto vale tudo isso em pessoas e em R$? | folha atual de 4.727 vai a ~4.967 pessoas (+240); acréscimo de ~R$ 30 milhões em 3 anos sobre a folha atual |

**O que mais limita o resultado não é a técnica, é o dado:** 76% do quadro operacional (toda a empresa 5 e os ônibus da empresa 8) não tem ponto nem afastamentos.

## 2. Os dados: o que encontramos antes de modelar

Os 14 CSVs têm o cadastro de 20.001 vínculos, o ponto diário (601.474 linhas) e 5.052 afastamentos. A auditoria (`notebooks/01_perfil_dos_dados.ipynb`) mostrou:

- **Cobertura parcial.** O ponto e os afastamentos vêm da **mesma subpopulação de ~1.900 pessoas**, quase só motoristas de caminhão da empresa 6. A empresa 5 (60% dos ativos) não tem nenhuma linha de ponto. Entre os ônibus, só 46 de 1.972 motoristas têm ponto.
- **Cadastro longo.** Admissões e demissões cobrem 2016–2026, bem além dos 21 meses do ponto. A Trilha 1 usa isso.
- **Artefatos de dados que distorceriam as taxas**, todos tratados e documentados:
  - nas empresas 3 e 8, o ponto só registra batidas depois de ~20/01/2025 e ~20/03/2025 (antes disso, ~100% de falta);
  - 6 dias de abono coletivo (Carnaval, 24 e 31/12);
  - horários genéricos de cadastro: 1.194 tripulações da empresa 5 "em 5x2" com o mesmo horário de segunda a sexta, e 803 desligamentos em massa da empresa 6 com o mesmo horário.
- Um vínculo com demissão anterior à admissão, 150 batidas sem cadastro e 901 pessoas com mais de um vínculo (recontratação).

## 3. Trilha 1: rotatividade

**Como foi feito** (`notebooks/02_trilha1_rotatividade.ipynb`, `src/rotatividade.py`):

1. Série mensal de admissões, rescisões e headcount por empresa, de 2016 a set/2026, e turnover pela fórmula do enunciado.
2. Diagnóstico: sazonalidade fraca e sem padrão comum, e forte efeito do **tempo de casa** (na empresa 6, ~22% de saída no 3º mês contra ~1,7% com mais de 5 anos).
3. **Backtest de origem móvel**: projetar 12, 24 e 36 meses e comparar com o realizado, para 8 estratégias simples e para um modelo de coortes por tempo de casa.
4. Escolha: **média dos últimos 36 meses** por empresa, com faixa calibrada pelo backtest e limitada por um teto. Ela empata com as demais médias (erro de ~6,2 a ~6,9 p.p.), é a melhor no regime recente e a mais simples. O modelo de coortes **não superou** a média.

**Resultado** (turnover mensal anualizado, isto é, a fórmula do enunciado aplicada a cada mês × 12; faixa para os 3 anos):

| Empresa | Base | Faixa | Confiança |
|---|---|---|---|
| 5 | 10,2% | 10,0% a 12,8% | alta |
| 8 | 28,5% | 28,5% a 34,5% | média-alta |
| 6 | 61,1% | 61,1% a 71,0% | baixa |
| 3 | 32,8% | 20,7% a 41,1% | média-baixa |
| 7 | 13,6% | 13,6% a 24,5% | média |
| 4 | 27,0% | 20,4% a 46,4% | baixa (7 pessoas) |
| 1 | sem projeção (sem vínculos ativos) | | |

**Atenção à definição.** A mesma fórmula do enunciado aplicada a **um ano inteiro** dá valores menores, porque o headcount passa a incluir todos que passaram pela empresa no ano: empresa 6 = 42,5% em 2024 e 27,6% em 2025; empresa 8 = 22,4% e 21,9%; empresa 5 = 10,1% e 8,4% (`resultados/trilha1_turnover_definicoes.csv`). Aplicada à janela inteira de 36 meses (o horizonte do contrato), dá 60,9% na empresa 6, 47,6% na 8 e 23,4% na 5.

A empresa 6 é imprevisível (erro do backtest de 26 a 28 p.p. em 36 meses): o turnover dela depende de ganhar ou perder contratos.

## 4. Trilha 2: folguistas e feristas

**Como foi feito** (`notebooks/03_trilha2_folguistas_feristas.ipynb`, `src/ausencias.py`):

1. Perda de carga = fração do trabalho programado que não é entregue por ausência. Quadro extra por posição = `perda / (1 − perda)`.
2. A perda foi dividida em **três blocos**, cada um com seu tratamento:
   - **férias:** taxa entre quem tem 1 ano de casa ou mais (8,0% a 8,4%, próximo dos 30 dias por ano da lei, mas 23,6% dos dias marcados como férias no ponto estão em sequências de mais de 30 dias, o que indica afastamentos longos registrados como férias), multiplicada pela parcela do quadro com esse direito, projetada com o modelo de risco por tempo de casa da Trilha 1;
   - **afastamento prolongado** (INSS, acidente, licenças): nível médio;
   - **ausência curta imprevista** (faltas, atestados, abonos): nível médio mais uma **reserva** calibrada pela variabilidade diária, para 95% de nível de serviço (~2,5 p.p. acima da média).
3. Limpeza do ponto (seção 2 acima) e exclusão dos aprendizes (229 ativos, que têm 27% a 35% de abono por horas de estudo).
4. Células sem dado próprio usam **ponderação por credibilidade** `n/(n+100)` contra a referência geral.

**Resultado** (quadro extra em % das posições, ano 1 e ano 3):

| Empresa | Ano 1 | Ano 3 | Confiança |
|---|---|---|---|
| 5 | 21,6% | 21,7% | baixa (nenhum dado próprio) |
| 8 | 20,1% | 19,8% | média-baixa |
| 6 | 18,0% | 18,7% | alta |
| 3 | 14,8% | 16,6% | média |
| 7 | 32,6% | 33,8% | baixa |
| 4 | 27,1% | 32,7% | muito baixa |

**Validação** (seção 3b do notebook): estimando com 2025 e prevendo 2026, o método erra em média **1,2 p.p.** de perda (2,6 p.p. sem o ajuste de tempo de casa). Já **a extrapolação para células sem dado erra 5,4 p.p.** (RMSE), o que mostra que a faixa baixo–alto da Trilha 2 é estreita demais para a empresa 5.

## 5. Trilha 3: escala 6x1 para 5x2

**Como foi feito** (`notebooks/04_trilha3_escala.ipynb`, `src/escala.py`):

1. Folha nova / folha atual = horas semanais atuais / horas semanais novas.
2. Medimos no ponto o que cada escala realmente trabalha: **6x1 = 5,98 dias × 7h08 = 42,7 h/semana; 5x2 = 5,04 dias × 8h33 = 43,0 h/semana.** O 5x2 praticado hoje é a mesma carga semanal em 5 dias, não uma jornada menor.
3. Por isso o adicional depende de uma decisão de contrato, e foram modelados três cenários de jornada.
4. A rotatividade entra como sensibilidade, porque os dados não permitem estimar o efeito da escala sobre ela (a escala dos desligados é artefato de cadastro).

**Resultado** (integrantes a mais, para a folha em 6x1):

| Cenário de jornada | Fator de folha | População A (736 em 6x1 no cadastro) | População B (inclui 1.917 com escala desconhecida) |
|---|---|---|---|
| **Jornada diária mantida** (leitura do enunciado, caso base) | 1,197 | **145** (125 na empresa 6, 14 na 8, 6 na 7) | 522 |
| Limite semanal de 40 h | 1,10 | 73 | 265 |
| Carga semanal mantida | 1,00 | 0 | 0 |

A população B supõe que as tripulações de "jornada mensal" (empresa 5) e "revezamento" (empresa 8) operem em 6x1, o que os dados não confirmam. É um teto.

## 6. Modelo integrado e impacto em R$

**Como foi feito** (`notebooks/05_modelo_integrado.ipynb`, `src/integrado.py`). A base comum é a **posição** (a pessoa escalada num dia), e cada trilha vira um multiplicador:

`folha por posição = (1 + s·(f − 1)) / (1 − a) × (1 + T·dias_vaga/365)`

em que `a` é a perda por ausência (T2), `s` e `f` são a parcela em 6x1 e o fator de folha (T3), e `T` é o turnover (T1).

**Resultado:**

- **Por 100 posições**, a folha tem de ter 123 (empresa 5), 124 (empresa 8) e 145 (empresa 6) pessoas, no caso base.
- **Operação atual:** os 4.727 do quadro ativo se decompõem em 3.916 posições e 811 de cobertura. Com a mudança de escala e as vagas por rotatividade, a folha necessária vai a **4.967 (+240, ou +5,1%)**.
- **Custo** (salário de referência × 12): R$ 151 milhões por ano só de salário, ou R$ 257 milhões com encargos de 1,7. **O acréscimo sobre a folha atual é de cerca de R$ 10 milhões por ano, ou R$ 30 milhões em 3 anos** (+240 pessoas: mudança de escala R$ 17 milhões e vagas R$ 13 milhões em 3 anos). O custo da cobertura de ausências (R$ 128 milhões em 3 anos) **já está dentro da folha atual** e não é custo novo; somado aos outros dois, dá ~R$ 158 milhões acima do custo das posições puras.

## 7. As trilhas se relacionam?

Sim, e o modelo integrado torna isso explícito, mas a ligação tem graus diferentes:

- **T1 para T2 (forte, quantitativa):** o risco por tempo de casa define a parcela do quadro com direito a férias. Na empresa 6, isso reduz as férias de ~8% para ~4,6%. Validado: previsto 4,3% contra 4,6% observado.
- **T2 para T3 (composição):** a folha da T3 já contém a cobertura de ausências, então o fator de escala a multiplica.
- **T1 para T3 (sensibilidade):** a folha maior gera mais saídas e vagas, mas é efeito de segunda ordem.

Uma inconsistência foi encontrada e corrigida: a T2 tratava o quadro ativo como *posições* e a T3 como *folha*. No modelo integrado, a T2 passa a medir a cobertura dentro da folha (811 pessoas, e não 975).

## 8. O que mais pesa na incerteza

Quase tudo vem de **informação que a SIAN tem e nós não**:

1. **O ponto da empresa 5.** Sem ele, o erro de extrapolação das taxas de ausência é de ~±8 pessoas por 100 posições e custa cerca de **±R$ 11 milhões por ano**.
2. **A escala real das tripulações** das empresas 5 e 8 (jornada mensal e revezamento): muda o adicional de 145 para até 522.
3. **A jornada após a mudança de escala:** vai de 0 a +20% da folha em 6x1.

Duas premissas nossas, que não vêm dos dados e estão como parâmetros: **30 dias para repor uma saída** e **encargos de 1,7× sobre o salário**.

## 9. Limitações

- 76% do quadro operacional sem dado próprio (empresa 5 e ônibus da empresa 8).
- Taxas constantes por 3 anos, sem reajuste, inflação ou crescimento de quadro.
- A folha atual é tratada como adequadamente dimensionada.
- 13.211 dias sem batida, falta ou abono foram tratados como compensados (banco de horas). Se fossem faltas, a empresa 6 subiria de ~18% para ~25% na T2.
- Aprendizes ficam fora do dimensionamento.
- A faixa baixo–alto da T2 nos CSVs ainda é a estreita; o erro de extrapolação foi propagado só no notebook 05.

## 10. O que falta

- **Relatório em PDF de até 10 páginas** (obrigatório, a falta desclassifica) e envio por e-mail com o código, até 21/10 às 19h00.
- Corrigir a faixa da T2 para células sem dado, dentro da própria trilha.
- Dúvidas à organização: o que é a `META` em `dim_meta`, a escala real das tripulações das empresas 5 e 8, e a jornada pós-mudança.

## 10b. Correções de 21/10/2026 (revisão)

Os notebooks foram reexecutados do zero e reproduzem os CSVs. A revisão encontrou e corrigiu:

1. **Bug na simulação por tempo de casa** (`simular_coorte` e `participacao_elegivel`): quem tinha 199 meses de casa ou mais saía do quadro sem ser contado como desligamento. Agora o último compartimento é absorvente (`rotatividade.envelhecer`). Efeito: T2 da empresa 8 +0,4 p.p. (19,7% → 20,1%), empresas 6 e 7 até +0,5 p.p.; teto da faixa da T1 da empresa 6 de 72,1% para 71,0%. A projeção-base da T1, a T3 e o total de 4.967 pessoas não mudam.
2. **Definição do turnover:** o valor projetado é o turnover mensal anualizado. A fórmula aplicada a um ano inteiro dá valores menores; as duas estão em `resultados/trilha1_turnover_definicoes.csv`.
3. **Leitura do custo:** os R$ 158 milhões eram o custo acima das posições puras, não o aumento sobre a folha atual. O acréscimo real é de ~R$ 30 milhões em 3 anos (`resultados/integrado_acrescimo_folha_atual.csv`).
4. **Férias:** 23,6% dos dias `F` do ponto estão em sequências de mais de 30 dias (75 vínculos, máximo de 472 dias). A taxa próxima de 8,2% não valida o modelo de férias; o quadro total quase não muda, porque esses dias continuam sendo ausência.

## 11. Mapa dos arquivos

| Arquivo | Conteúdo |
|---|---|
| `docs/regras/REGRAS.md` | regras, datas e formato de envio |
| `docs/problema/PROBLEMA.md` | o que cada trilha pede e como se ligam |
| `notebooks/01_perfil_dos_dados.ipynb` | auditoria dos dados |
| `notebooks/02_trilha1_rotatividade.ipynb` | Trilha 1 |
| `notebooks/03_trilha2_folguistas_feristas.ipynb` | Trilha 2, com validação |
| `notebooks/04_trilha3_escala.ipynb` | Trilha 3 |
| `notebooks/05_modelo_integrado.ipynb` | modelo integrado e custos |
| `src/` | `rotatividade.py`, `ausencias.py`, `escala.py`, `integrado.py` |
| `resultados/` | CSVs com os resultados de cada trilha e do modelo integrado |
| `requirements.txt` | dependências (testadas em ambiente limpo) |
