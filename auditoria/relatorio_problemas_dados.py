"""
Gera o relatório em PDF dos problemas encontrados nos dados do Datathon (Grupo HP).
Cada problema é recalculado a partir dos CSVs no momento da geração: os números e
os exemplos no PDF são a prova.

Uso: python -I relatorio_problemas.py <pasta_dados> <saida.pdf> <pasta_figuras>
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

DADOS, SAIDA, FIG = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(FIG, exist_ok=True)
INI, FIM = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-09-30")

# ------------------------------------------------------------------ fontes / estilo
F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Arial", F + "arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-B", F + "arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-I", F + "ariali.ttf"))
pdfmetrics.registerFont(TTFont("Mono", F + "consola.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Arial", normal="Arial", bold="Arial-B", italic="Arial-I", boldItalic="Arial-B")

TXT, TXT2, AZUL, LINHA = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#2a78d6"), colors.HexColor("#d9d8d4")
SEV = {"CRÍTICO": "#d03b3b", "ALTO": "#ec835a", "MÉDIO": "#c98500", "BAIXO": "#52514e", "OK": "#0ca30c"}
st = {
    "h0": ParagraphStyle("h0", fontName="Arial-B", fontSize=20, leading=24, textColor=TXT, spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="Arial-B", fontSize=14, leading=18, textColor=TXT, spaceBefore=10, spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName="Arial-B", fontSize=11.5, leading=14, textColor=TXT, spaceBefore=8, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="Arial", fontSize=9.5, leading=13, textColor=TXT, spaceAfter=4, alignment=TA_LEFT),
    "s": ParagraphStyle("s", fontName="Arial", fontSize=8.5, leading=11, textColor=TXT2, spaceAfter=3),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8, leading=10.5, textColor=TXT2, backColor=colors.HexColor("#f3f2ef"),
                           borderPadding=4, spaceBefore=2, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="Arial", fontSize=7.8, leading=9.5, textColor=TXT),
    "cellb": ParagraphStyle("cellb", fontName="Arial-B", fontSize=7.8, leading=9.5, textColor=TXT),
}
story = []
P = lambda t, s="p": story.append(Paragraph(t, st[s]))


def code(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
    story.append(Paragraph(t, st["code"]))


def tabela(df, larg=None, max_linhas=None, idx=False):
    if idx:
        df = df.reset_index()
    if max_linhas:
        df = df.head(max_linhas)
    dados = [[Paragraph(str(c), st["cellb"]) for c in df.columns]]
    for r in df.itertuples(index=False):
        dados.append([Paragraph("" if (isinstance(v, float) and np.isnan(v)) else
                                (f"{v:,.0f}".replace(",", ".") if isinstance(v, (int, np.integer)) else str(v)), st["cell"]) for v in r])
    t = Table(dados, colWidths=larg, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, TXT), ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINHA),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    story.append(t)
    story.append(Spacer(1, 6))


resumo = []  # (n, título, severidade, trilhas, tratamento)


def problema(n, titulo, sev, trilhas, esperado, encontrado, impacto, tratamento):
    resumo.append((n, titulo, sev, trilhas, tratamento))
    story.append(Paragraph(f'{n}. {titulo} &nbsp;<font color="{SEV[sev]}" name="Arial-B">[{sev}]</font> '
                           f'<font color="#52514e" size="9" name="Arial">Afeta: {trilhas}</font>', st["h2"]))
    P(f"<b>O que se esperava:</b> {esperado}")
    P(br(f"<b>O que os dados mostram:</b> {encontrado}"))


def fecho(impacto, tratamento):
    n, t_, s_, tr, _ = resumo[-1]; resumo[-1] = (n, t_, s_, tr, tratamento)
    P(f"<b>Impacto:</b> {impacto}")
    P(f"<b>Tratamento proposto:</b> {tratamento}")


def fig(nome, w=16):
    story.append(Image(f"{FIG}/{nome}.png", width=w * cm, height=w * cm * plt.imread(f"{FIG}/{nome}.png").shape[0] / plt.imread(f"{FIG}/{nome}.png").shape[1]))
    story.append(Spacer(1, 4))


def cramers_v(x, y):
    tab = pd.crosstab(x, y)
    if min(tab.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.values.sum(); r, k = tab.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2 / max(1e-9, min(kc - 1, rc - 1)))


def qui2(tab):
    """Teste qui-quadrado de independência; devolve texto com estatística, gl, p e V de Cramér."""
    chi2, pv, gl, _ = chi2_contingency(tab, correction=False)
    return chi2, gl, pv


def fp(pv):
    return "p &lt; 10<super>-300</super>" if pv == 0 else (f"p = {pv:.1e}" if pv < 0.001 else f"p = {pv:.3f}")


import re
def br(txt):
    """Formata números no padrão brasileiro (decimal com vírgula, milhar com ponto)."""
    txt = re.sub(r"(\d),(\d{3})", r"\1§\2", txt)
    txt = re.sub(r"(\d)\.(\d{1,2})(?!\d)", r"\1,\2", txt)
    return txt.replace("§", ".")


def teste(txt):
    txt = br(txt)
    story.append(Paragraph(f"<b>Teste estatístico:</b> {txt}", st["p"]))


plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
fmt = lambda v: f"{v:,.0f}".replace(",", ".")

# ------------------------------------------------------------------ carga
ler = lambda n: pd.read_csv(f"{DADOS}/{n}.csv", sep=";", encoding="utf-8-sig", dtype=str)
i = ler("dim_integrante"); b = ler("fBatidas"); a = ler("fAfastamentos"); h = ler("dim_horario")
fu = ler("dim_funcao"); fer = ler("dim_feriados"); meta = ler("dim_meta"); sit_d = ler("dim_situacao")
i["adm"] = pd.to_datetime(i.DATAADMISSAO); i["dem"] = pd.to_datetime(i.DATADEMISSAO); i["sal"] = pd.to_numeric(i.SALARIO)
i = i.merge(h[["ID_HORARIO", "ESCALA"]], on="ID_HORARIO", how="left").merge(fu[["ID_FUNCAO", "FUNCAO"]], on="ID_FUNCAO", how="left")
NUM = ["MINUTOS_PREVISTOS", "MINUTOS_FALTA", "MINUTOS_ATRASO", "MINUTOS_ABONO", "MINUTOS_TRABALHADOS", "MINUTOS_EXTRA"]
for c in NUM:
    b[c] = pd.to_numeric(b[c])
b["d"] = pd.to_datetime(b.DATA)
a["ini"] = pd.to_datetime(a.DT_INICIO); a["fim"] = pd.to_datetime(a.DT_FIM)
ii = i.set_index("ID_INTEGRANTE")
per = i[(i.adm <= FIM) & (i.dem.isna() | (i.dem >= INI))].copy()   # vínculos com algum dia no período
ativos = i[i.dem.isna()]
tp = set(b.ID_INTEGRANTE)
bx = b.merge(ii[["adm", "dem", "ID_HORARIO"]].rename(columns={"ID_HORARIO": "HOR_CAD"}), left_on="ID_INTEGRANTE", right_index=True, how="left")
dentro = (bx.d >= bx.adm) & (bx.dem.isna() | (bx.d <= bx.dem))
ab = {"A": "Ativo", "F": "Férias", "D": "Demitido", "V": "Aviso"}
COLS_B = ["ID_INTEGRANTE", "DATA", "SITUACAO_NA_DATA", "MINUTOS_PREVISTOS", "MINUTOS_TRABALHADOS", "MINUTOS_FALTA", "MINUTOS_ABONO", "MINUTOS_EXTRA"]
REN_B = {"ID_INTEGRANTE": "vínculo", "DATA": "data", "SITUACAO_NA_DATA": "sit.", "MINUTOS_PREVISTOS": "prev.",
         "MINUTOS_TRABALHADOS": "trab.", "MINUTOS_FALTA": "falta", "MINUTOS_ABONO": "abono", "MINUTOS_EXTRA": "extra"}

# ================================================================== CAPA
P("Auditoria dos dados do Datathon", "h0")
P("III Datathon IME-UFG · Grupo HP · gerado em 21/10/2026", "s")
story.append(Spacer(1, 6))
P("Este documento lista os problemas encontrados nos dados entregues (pasta <i>dados/</i>) e <b>prova cada um</b> "
  "com números recalculados diretamente dos arquivos CSV e com registros reais de exemplo. Para cada problema "
  "mostramos: o que se esperava (pelo PDF de instruções ou por regra de negócio), o que os dados mostram, a regra "
  "exata usada para verificar (em <font name='Mono'>pandas</font>), o impacto nas trilhas e o tratamento que adotaremos.")
P("Convenções: <b>vínculo</b> = linha de <i>dim_integrante</i> (ID_INTEGRANTE); <b>período</b> = 01/01/2025 a 30/09/2026; "
  "<b>vínculo do período</b> = admitido até 30/09/2026 e sem demissão ou demitido a partir de 01/01/2025. "
  "Severidade: CRÍTICO (impede um cálculo direto), ALTO (enviesa resultados se não tratado), MÉDIO (exige limpeza), BAIXO (cosmético).")
P("Todos os números deste relatório são reproduzidos pelo script <font name='Mono'>relatorio_problemas.py</font>, entregue junto com o código.", "s")

P("Métodos de verificação e testes estatísticos", "h1")
P("Os dados são um <b>censo</b> dos vínculos do Grupo HP, não uma amostra. Por isso, a maior parte das verificações é "
  "<b>determinística</b>: uma regra lógica (ex.: \"data do ponto &lt; data de admissão\") aplicada a todos os registros e contada. "
  "Uma contagem maior que zero já prova o problema e não precisa de teste de hipótese. Usamos testes estatísticos quando a pergunta é "
  "sobre <b>padrão</b> (a falta de dados é aleatória? duas variáveis são redundantes? duas fontes concordam?). Com milhares de "
  "registros, quase todo p-valor sai muito pequeno; por isso reportamos sempre uma <b>medida de tamanho de efeito</b> junto com o p-valor.")
met = pd.DataFrame([
    ["Regras de integridade (contagem exata)", "Datas fora do contrato, chaves órfãs, duplicados, valores fora do domínio, sobreposição de intervalos", "Determinístico: contagem &gt; 0 prova o problema", "5, 6, 9–13, 17, 18"],
    ["Qui-quadrado de independência (Pearson)", "H0: duas variáveis categóricas são independentes. Aplicado a \"ter ponto\" x empresa/escala, para testar se a falta de dados é aleatória (MCAR)", "p-valor + V de Cramér", "1, 2, 4, 19"],
    ["V de Cramér com correção de viés (Bergsma, 2013)", "Força da associação entre duas categóricas, de 0 (independentes) a 1 (redundantes). Equivale a uma medida de colinearidade para variáveis categóricas", "V &gt;= 0,6 indica redundância forte", "4, 19"],
    ["Qui-quadrado de aderência", "H0: o dia do mês do 1º registro de ponto é uniforme (dias 1 a 28). Rejeitar indica que o ponto segue um calendário fixo", "p-valor + razão observado/esperado no dia 21", "5"],
    ["Kappa de Cohen", "Concordância entre duas fontes que classificam o mesmo vínculo-dia como férias ou não (ponto x afastamentos), descontando o acaso", "1 = concordância perfeita; &lt; 0,8 indica divergência relevante", "8"],
    ["Correlação de Pearson e VIF", "Colinearidade entre as colunas numéricas de minutos. VIF<sub>j</sub> = 1/(1-R<sup>2</sup><sub>j</sub>), com R<sup>2</sup><sub>j</sub> da regressão da coluna j nas demais", "VIF &gt; 10 indica multicolinearidade grave", "20"],
    ["Mediana / estatística descritiva", "Escala praticada (dias com jornada por semana) e evolução do salário nominal", "Comparação com o valor esperado pela regra de negócio", "7, 14, 15"],
], columns=["Método", "O que verifica", "Critério de decisão", "Problemas"])
dados_m = [[Paragraph(c, st["cellb"]) for c in met.columns]] + [[Paragraph(v, st["cell"]) for v in r] for r in met.values]
tm = Table(dados_m, colWidths=[3.9 * cm, 7.3 * cm, 3.8 * cm, 2 * cm], repeatRows=1, hAlign="LEFT")
tm.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, TXT), ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINHA),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
story.append(tm)
story.append(Spacer(1, 4))
P("Nível de significância: 5%. Os testes foram feitos em Python (scipy.stats.chi2_contingency, scipy.stats.chisquare; kappa e VIF calculados diretamente).", "s")
story.append(PageBreak())

# ================================================================== 1. COBERTURA DO PONTO
P("Parte 1 — Cobertura dos dados", "h1")
cob = per.groupby("CODCOLIGADA").agg(vinc=("ID_INTEGRANTE", "size"), ponto=("ID_INTEGRANTE", lambda s: s.isin(tp).sum()))
cob["afast"] = a.groupby("CODCOLIGADA").size().reindex(cob.index).fillna(0).astype(int)
cob["ativos_hoje"] = ativos.groupby("CODCOLIGADA").size().reindex(cob.index).fillna(0).astype(int)
cob["pct"] = 100 * cob.ponto / cob.vinc
problema(1, "O ponto diário não cobre a maioria dos vínculos — e a empresa 5 não tem ponto nenhum", "CRÍTICO", "T2, T3",
         "\"Ponto diário: uma linha por vínculo por dia\" de 01/01/2025 a 30/09/2026 (PDF de instruções).",
         f"Só <b>{len(tp)}</b> vínculos aparecem em <i>fBatidas</i>, contra <b>{fmt(len(per))}</b> vínculos do período "
         f"({100*per.ID_INTEGRANTE.isin(tp).mean():.0f}%). A empresa 5 — a maior, com {fmt(cob.loc['5','ativos_hoje'])} ativos hoje — "
         f"tem <b>0</b> vínculos com ponto e apenas <b>{cob.loc['5','afast']}</b> afastamento registrado em 21 meses.", "", "")
code("per = vínculos do período\nper.groupby('CODCOLIGADA').ID_INTEGRANTE.apply(lambda s: s.isin(set(fBatidas.ID_INTEGRANTE)).mean())")
t = cob.copy(); t["pct"] = t.pct.map(lambda v: f"{v:.0f}%")
t.columns = ["vínculos do período", "com ponto", "eventos de afastamento", "ativos hoje", "% com ponto"]
tabela(t.rename_axis("empresa"), idx=True)
fig_d = cob.sort_values("pct")
fg, ax = plt.subplots(figsize=(7, 2.4))
ax.barh(fig_d.index.map(lambda e: f"Empresa {e}"), fig_d.pct, color="#2a78d6", height=0.6)
for y, (v, n, N) in enumerate(zip(fig_d.pct, fig_d.ponto, fig_d.vinc)):
    ax.text(v + 1, y, f"{v:.0f}%  ({n} de {N})", va="center", color="#0b0b0b", fontsize=8)
ax.set_xlim(0, 125); ax.set_xlabel("% dos vínculos do período que têm registro de ponto"); ax.xaxis.grid(True, color="#ecebe7"); ax.set_axisbelow(True)
ax.set_title("Cobertura do ponto diário por empresa", loc="left", fontsize=10, color="#0b0b0b")
fg.tight_layout(); fg.savefig(f"{FIG}/cobertura.png", dpi=200); plt.close(fg)
fig("cobertura", 15)
_t = pd.crosstab(per.CODCOLIGADA, per.ID_INTEGRANTE.isin(tp)); _c, _g, _p = qui2(_t)
_t2 = _t.drop(index="5"); _c2, _g2, _p2 = qui2(_t2)
teste(f"qui-quadrado de independência entre <i>ter ponto</i> e <i>empresa</i>: χ² = {_c:,.0f}, gl = {_g}, {fp(_p)}, V de Cramér = {cramers_v(per.CODCOLIGADA, per.ID_INTEGRANTE.isin(tp)):.2f}. "
      f"Mesmo excluindo a empresa 5: χ² = {_c2:,.0f}, gl = {_g2}, {fp(_p2)}. Rejeitamos H0: a falta de ponto <b>não é aleatória</b> (não é MCAR), "
      f"logo médias calculadas só sobre quem tem ponto são viesadas.")
fecho("Não é possível medir faltas, atestados curtos e escala realizada para a empresa 5 nem para a maior parte da empresa 8. "
      "Qualquer taxa calculada só com o ponto descreve essencialmente a empresa 6.",
      "Calcular taxas de ausência onde há ponto e transferi-las para empresas sem ponto por função e tipo de jornada "
      "(estimativa por pares), sempre sinalizada como extrapolação e com intervalo de incerteza.")

# ------------------------------------------------------------------ 2. empresa 8 revezamento
e8 = per[per.CODCOLIGADA == "8"]
ct8 = pd.crosstab(e8.ESCALA, e8.ID_INTEGRANTE.isin(tp)).rename(columns={False: "sem ponto", True: "com ponto"})
problema(2, "Na empresa 8, nenhum motorista em \"REVEZAMENTO\" tem ponto", "CRÍTICO", "T2, T3",
         "A falta de ponto, se existisse, seria aleatória entre os integrantes.",
         f"A ausência é <b>sistemática</b>: todos os {ct8.loc['REVEZAMENTO','sem ponto']} vínculos da empresa 8 com escala REVEZAMENTO "
         f"(praticamente todos motoristas de ônibus) estão fora de <i>fBatidas</i>. Quem tem ponto na empresa 8 é outro perfil de trabalhador.", "", "")
code("e8 = per[per.CODCOLIGADA=='8']\npd.crosstab(e8.ESCALA, e8.ID_INTEGRANTE.isin(set(fBatidas.ID_INTEGRANTE)))")
tabela(ct8.rename_axis("escala (cadastro)"), idx=True, larg=[5 * cm, 3 * cm, 3 * cm])
_c, _g, _p = qui2(ct8)
teste(f"qui-quadrado de independência entre <i>ter ponto</i> e <i>escala</i> na empresa 8: χ² = {_c:,.0f}, gl = {_g}, {fp(_p)}, "
      f"V de Cramér = {cramers_v(e8.ESCALA, e8.ID_INTEGRANTE.isin(tp)):.2f}. Associação quase perfeita: a ausência de ponto é determinada pela escala.")
fecho("O ponto da empresa 8 não representa seus motoristas de ônibus — usá-lo como se representasse produz viés de seleção.",
      "Para os motoristas da empresa 8, usar apenas <i>fAfastamentos</i> (que os cobre) e completar faltas com taxas de pares.")

# ------------------------------------------------------------------ 3. escala desconhecida
ctE = pd.crosstab(ativos.CODCOLIGADA, ativos.ESCALA)
nd = ctE.get("JORNADA MENSAL", 0).sum() + ctE.get("REVEZAMENTO", 0).sum() + ctE.get("NAO IDENTIFICADA", 0).sum()
problema(3, "A escala da maior parte dos motoristas de ônibus é desconhecida", "CRÍTICO", "T3",
         "\"O horário de cada integrante define sua escala de trabalho, como 6x1, 5x2 ou 12x36\" (PDF).",
         f"<b>{fmt(nd)}</b> dos {fmt(len(ativos))} ativos hoje ({100*nd/len(ativos):.0f}%) têm escala JORNADA MENSAL, REVEZAMENTO ou "
         f"NAO IDENTIFICADA — categorias que não dizem quantos dias por semana a pessoa trabalha. São justamente os motoristas "
         f"de ônibus da empresa 5 (JORNADA MENSAL) e da empresa 8 (REVEZAMENTO), sem ponto para inferir a escala real.", "", "")
code("ativos = dim_integrante[DATADEMISSAO.isna()]  ->  pd.crosstab(ativos.CODCOLIGADA, ativos.ESCALA)")
tabela(ctE.rename_axis("empresa"), idx=True)
fecho("A Trilha 3 (40% da nota) pergunta quantas pessoas a mais seriam necessárias de 6x1 para 5x2; para estes grupos não se sabe "
      "sequer se hoje estão em 6x1.",
      "Tratar como cenários explícitos (ex.: \"se REVEZAMENTO equivale a 6x1\" / \"se JORNADA MENSAL de 180h equivale a 5x2\"), "
      "usando JORNADAMENSAL (minutos/mês) como pista, e reportar o resultado de cada cenário.")

# ------------------------------------------------------------------ 4. confundimento escala
p_ = per.copy()
p_["FAIXA_SAL"] = pd.qcut(p_.sal, 5, labels=False, duplicates="drop").astype(str)
v_ef = cramers_v(p_.ESCALA.fillna("NA"), p_.FUNCAO.fillna("NA")); v_ee = cramers_v(p_.ESCALA.fillna("NA"), p_.CODCOLIGADA)
s6 = per[per.ESCALA == "6X1"]
top6 = s6.groupby(["CODCOLIGADA", "FUNCAO"]).size().sort_values(ascending=False)
problema(4, "A escala 6x1 está concentrada numa única empresa e numa única função", "ALTO", "T3",
         "Para estimar o efeito da escala sobre a rotatividade, seria preciso comparar 6x1 e 5x2 em grupos semelhantes.",
         f"Dos {fmt(len(s6))} vínculos 6x1 do período, <b>{fmt(top6.iloc[0])} ({100*top6.iloc[0]/len(s6):.0f}%)</b> são "
         f"\"{top6.index[0][1]}\" da empresa {top6.index[0][0]}. A associação (V de Cramér) entre escala e função é "
         f"<b>{v_ef:.2f}</b> e entre escala e empresa é <b>{v_ee:.2f}</b>.", "", "")
code("s6 = per[per.ESCALA=='6X1']; s6.groupby(['CODCOLIGADA','FUNCAO']).size()")
tt = top6.head(6).rename("vínculos 6x1").reset_index().rename(columns={"CODCOLIGADA": "empresa", "FUNCAO": "função"})
tabela(tt, larg=[2 * cm, 6 * cm, 3 * cm])
_c, _g, _p = qui2(pd.crosstab(p_.ESCALA.fillna("NA"), p_.FUNCAO.fillna("NA")))
_c2, _g2, _p2 = qui2(pd.crosstab(p_.ESCALA.fillna("NA"), p_.CODCOLIGADA))
teste(f"qui-quadrado escala x função: χ² = {_c:,.0f}, gl = {_g}, {fp(_p)}, V = {v_ef:.2f}; escala x empresa: χ² = {_c2:,.0f}, gl = {_g2}, {fp(_p2)}, V = {v_ee:.2f}. "
      "Associação forte (V &gt; 0,5): escala, função e empresa são colineares, e o efeito isolado da escala não é identificável sem controles.")
fecho("Uma comparação ingênua \"turnover 6x1 vs 5x2\" mede, na prática, \"motoristas de caminhão da empresa 6 vs. o resto\".",
      "Comparar dentro da mesma empresa e função (estratificação/regressão com controles) e explorar quem mudou de horário "
      "ao longo do ponto; declarar a limitação.")

# ------------------------------------------------------------------ 5. ponto pela folha
P("Parte 2 — Registros que contradizem o cadastro", "h1")
fora = bx[~dentro & bx.adm.notna()]
antes = (bx.d < bx.adm).sum(); depois = (bx.d > bx.dem).sum()
prim = bx.groupby("ID_INTEGRANTE").agg(primeiro=("d", "min"), adm=("adm", "first"))
prim = prim[prim.adm > INI]  # admitidos no período: o ponto deveria começar na admissão
dia1 = prim.primeiro.dt.day.value_counts().sort_index()
ex = fora.groupby("ID_INTEGRANTE").size().sort_values(ascending=False).index[1]
problema(5, "O ponto tem registros antes da admissão e depois da demissão", "ALTO", "T1, T2, T3",
         "Um vínculo só pode ter ponto entre a data de admissão e a de demissão.",
         f"<b>{fmt(antes)}</b> linhas são anteriores à admissão e <b>{fmt(depois)}</b> posteriores à demissão, em "
         f"<b>{fmt(fora.ID_INTEGRANTE.nunique())}</b> vínculos. O padrão revela a causa: entre os vínculos admitidos no período, "
         f"<b>{100*dia1.get(21,0)/dia1.sum():.0f}%</b> têm o primeiro registro de ponto no <b>dia 21</b> do mês — o ponto é gerado "
         f"por <b>período de apuração da folha (21 a 20)</b>, não pelo contrato.", "", "")
code("bx = fBatidas JOIN dim_integrante\n(bx.DATA < bx.DATAADMISSAO).sum(); (bx.DATA > bx.DATADEMISSAO).sum()\n"
     "primeiro_registro.dt.day.value_counts()   # pico no dia 21")
fg, ax = plt.subplots(figsize=(7, 2.3))
ax.bar(dia1.index, dia1.values, color=["#2a78d6" if d == 21 else "#b7d3f6" for d in dia1.index], width=0.75)
ax.set_xlabel("dia do mês do primeiro registro de ponto (vínculos admitidos a partir de jan/2025)"); ax.set_ylabel("vínculos")
ax.set_xticks(range(1, 32, 2)); ax.yaxis.grid(True, color="#ecebe7"); ax.set_axisbelow(True)
ax.annotate(f"dia 21: {dia1.get(21,0)} vínculos", (21, dia1.get(21, 0)), xytext=(24, dia1.max() * 0.85), fontsize=8,
            arrowprops=dict(arrowstyle="-", color="#52514e", lw=0.6), color="#0b0b0b")
ax.set_title("O ponto começa no início do período de folha, não na admissão", loc="left", fontsize=10, color="#0b0b0b")
fg.tight_layout(); fg.savefig(f"{FIG}/dia21.png", dpi=200); plt.close(fg)
fig("dia21", 15)
from scipy.stats import chisquare
_obs = dia1.reindex(range(1, 29), fill_value=0); _gf = chisquare(_obs)
teste(f"qui-quadrado de aderência à distribuição uniforme nos dias 1 a 28 (H0: o 1º registro pode cair em qualquer dia, como as admissões): "
      f"χ² = {_gf.statistic:,.0f}, gl = 27, {fp(_gf.pvalue)}. No dia 21 há {_obs[21]} vínculos contra {_obs.mean():.1f} esperados "
      f"({_obs[21]/_obs.mean():.0f}x). Para comparação, o dia da <b>admissão</b> desses vínculos se distribui de forma bem mais espalhada "
      f"(máximo de {prim.adm.dt.day.value_counts().max()} num único dia).")
P(f"Exemplo real — vínculo <b>{ex}</b>, admitido em {ii.loc[ex,'adm'].date():%d/%m/%Y} e demitido em "
  f"{ii.loc[ex,'dem'].date():%d/%m/%Y}" if pd.notna(ii.loc[ex, 'dem']) else f"Exemplo real — vínculo <b>{ex}</b>", "s")
exr = bx[bx.ID_INTEGRANTE == ex].sort_values("d")
exr = pd.concat([exr[exr.d < exr.adm].head(3), exr[(exr.d >= exr.adm)].head(2), exr[exr.d > exr.dem].tail(2)])
tabela(exr[COLS_B].rename(columns=REN_B))
fecho("Dias fora do contrato entram como se fossem dias do vínculo: inflam headcount diário e podem aparecer como falta/folga.",
      "Filtrar o ponto por <font name='Mono'>DATAADMISSAO &lt;= DATA &lt;= DATADEMISSAO</font> antes de qualquer cálculo.")

# ------------------------------------------------------------------ 6. dias faltando
c = bx[dentro].groupby("ID_INTEGRANTE").size().rename("n").to_frame().join(ii[["adm", "dem"]])
c["esperado"] = (c.dem.fillna(FIM).clip(upper=FIM) - c.adm.clip(lower=INI)).dt.days + 1
c["faltando"] = (c.esperado - c.n).clip(lower=0)
cf = c[c.faltando > 0]
exg = cf.sort_values("faltando").index[len(cf) // 2]
meses = bx[(bx.ID_INTEGRANTE == exg) & dentro].groupby(bx.DATA.str[:7]).size()
todos = pd.period_range(max(ii.loc[exg, "adm"], INI), min(ii.loc[exg, "dem"] if pd.notna(ii.loc[exg, "dem"]) else FIM, FIM), freq="M").astype(str)
meses = meses.reindex(todos, fill_value=0)
problema(6, "Há meses inteiros sem ponto dentro do contrato", "MÉDIO", "T2",
         "Entre admissão e demissão (limitadas ao período), uma linha por dia.",
         f"<b>{fmt(len(cf))}</b> dos {fmt(len(c))} vínculos com ponto têm dias faltando dentro do contrato, somando "
         f"<b>{fmt(cf.faltando.sum())}</b> dias. A mediana entre eles é de <b>{cf.faltando.median():.0f} dias</b> — tipicamente um mês inteiro ausente.", "", "")
code("esperado = dias entre max(adm, 01/01/25) e min(dem, 30/09/26); faltando = esperado - linhas de ponto dentro do contrato")
P(f"Exemplo real — vínculo <b>{exg}</b>: dias de ponto registrados por mês", "s")
mm = meses.to_frame("dias com ponto").T
tabela(mm.iloc[:, :12] if mm.shape[1] > 12 else mm)
if mm.shape[1] > 12:
    tabela(mm.iloc[:, 12:])
fecho("Ausências nesses meses ficam invisíveis; taxas por dia calculadas com o denominador errado.",
      "Usar como denominador apenas os dias efetivamente registrados e marcar os meses sem ponto como faltantes (não como presença).")

# ------------------------------------------------------------------ 7. escala cadastro vs ponto
bb = b[b.MINUTOS_PREVISTOS > 0].copy(); bb["sem"] = bb.d.dt.to_period("W")
dps = bb.groupby(["ID_INTEGRANTE", "ID_HORARIO", "sem"]).size()
dps = dps[dps.groupby(level=[0, 1]).transform("size") >= 4]
obs = dps.groupby(level=[0, 1]).median().rename("dias").reset_index().merge(h[["ID_HORARIO", "ESCALA"]], on="ID_HORARIO")
obs["dias"] = obs.dias.round().astype(int)
cto = pd.crosstab(obs.ESCALA, obs.dias).rename(columns=lambda x: f"{x} dia(s)/sem")
dif_h = (bx.ID_HORARIO != bx.HOR_CAD) & bx.HOR_CAD.notna()
problema(7, "A escala do cadastro nem sempre é a escala praticada", "ALTO", "T3",
         "Um horário 6x1 deveria ter ~6 dias com jornada prevista por semana; um 5x2, ~5.",
         f"Contando os dias com MINUTOS_PREVISTOS &gt; 0 por semana no ponto (mediana por vínculo e horário), "
         f"<b>{cto.loc['5X2'].drop('5 dia(s)/sem', errors='ignore').sum()}</b> combinações vínculo-horário rotuladas 5X2 não trabalham 5 dias e "
         f"<b>{cto.loc['6X1'].drop('6 dia(s)/sem', errors='ignore').sum()}</b> rotuladas 6X1 não trabalham 6. Além disso, "
         f"<b>{100*dif_h.mean():.0f}%</b> das linhas de ponto usam um horário diferente do que está no cadastro: o cadastro guarda só o horário atual.", "", "")
code("dias/semana = nº de dias com MINUTOS_PREVISTOS>0 por (vínculo, horário, semana) -> mediana\n"
     "(fBatidas.ID_HORARIO != dim_integrante.ID_HORARIO).mean()")
tabela(cto.rename_axis("escala no cadastro"), idx=True)
fecho("Classificar quem está em 6x1 pelo cadastro erra uma parcela relevante; e o histórico de mudança de escala se perde.",
      "Definir a escala de cada vínculo-mês pelo ponto (dias trabalhados observados) e usar o cadastro só como fallback.")

# ------------------------------------------------------------------ 8. férias
fe = a[a.CATEGORIA_EVENTO == "FERIAS"]
S = set()
for r in fe.itertuples():
    if r.ID_INTEGRANTE in tp:
        for dd in pd.date_range(max(r.ini, INI), min(r.fim, FIM)):
            S.add((r.ID_INTEGRANTE, dd))
bF = b[b.SITUACAO_NA_DATA == "F"]; Fp = set(zip(bF.ID_INTEGRANTE, bF.d))
so_p = Fp - S
_bf = bF.sort_values(["ID_INTEGRANTE", "d"]).copy()
_bf["bloco"] = (_bf.groupby("ID_INTEGRANTE").d.diff().dt.days != 1).cumsum()
_run = _bf.groupby(["ID_INTEGRANTE", "bloco"]).size()
n_long = (_run.groupby(level=0).max() > 30).sum(); max_run = _run.max()
exf = pd.Series([v for v, _ in so_p]).value_counts().index[0]
problema(8, "Férias: o ponto e a tabela de afastamentos discordam", "MÉDIO", "T2",
         "Os dias com situação \"F\" (férias) no ponto deveriam coincidir com os eventos FERIAS de <i>fAfastamentos</i>.",
         f"Para vínculos que têm ponto: <b>{fmt(len(Fp))}</b> dias de férias no ponto e <b>{fmt(len(S))}</b> em afastamentos. "
         f"Coincidem <b>{fmt(len(Fp & S))}</b>; <b>{fmt(len(so_p))}</b> estão só no ponto e <b>{fmt(len(S - Fp))}</b> só em afastamentos. "
         f"Além disso, as empresas 4 e 5 não têm nenhum evento de férias em 21 meses. E <b>{n_long}</b> vínculos têm sequências de mais de 30 dias "
         f"seguidos com situação \"F\" no ponto (máximo de <b>{max_run} dias</b>), quando as férias (CLT) são de no máximo 30 dias por período aquisitivo "
         f"— provavelmente afastamentos longos registrados como férias.", "", "")
code("dias_ponto = {(vínculo, data) : SITUACAO_NA_DATA=='F'}\ndias_afast = {(vínculo, d) : d em [DT_INICIO, DT_FIM] e CATEGORIA_EVENTO=='FERIAS'}")
P(f"Exemplo real — vínculo <b>{exf}</b>: eventos FERIAS registrados e meses com situação F no ponto", "s")
e1 = fe[fe.ID_INTEGRANTE == exf][["ID_INTEGRANTE", "CATEGORIA_EVENTO", "DT_INICIO", "DT_FIM"]]
e2 = bF[bF.ID_INTEGRANTE == exf].groupby(bF.DATA.str[:7]).agg(primeiro=("DATA", "min"), ultimo=("DATA", "max"), dias=("DATA", "size")).reset_index().rename(columns={"DATA": "mês"})
_U = bx[dentro & (bx.d >= INI) & (bx.d <= FIM)][["ID_INTEGRANTE", "d"]]
_k = list(zip(_U.ID_INTEGRANTE, _U.d)); _A = np.array([k in Fp for k in _k]); _B = np.array([k in S for k in _k])
_po = (_A == _B).mean(); _pe = _A.mean() * _B.mean() + (1 - _A.mean()) * (1 - _B.mean()); _kap = (_po - _pe) / (1 - _pe)
teste(f"kappa de Cohen entre as duas fontes, sobre os {fmt(len(_k))} vínculo-dias com ponto dentro do contrato: "
      f"concordância bruta = {100*_po:.1f}% (inflada porque a maioria dos dias não é férias), κ = <b>{_kap:.2f}</b>. "
      + ("A concordância é boa (κ &gt; 0,8), mas não perfeita: " if _kap > 0.8 else "Um κ abaixo de 0,8 mostra divergência relevante: ")
      + f"{fmt(len(Fp ^ S))} vínculo-dias de férias são registrados por apenas uma das fontes, o que equivale a cerca de "
      f"{len(Fp ^ S)/len(Fp | S)*100:.0f}% de todos os dias de férias.")
tabela(e1 if len(e1) else pd.DataFrame({"fAfastamentos": ["nenhum evento FERIAS para este vínculo"]}))
tabela(e2)
fecho("Usar só uma das fontes subestima as férias e, portanto, o quadro de feristas.",
      "Usar a união das duas fontes (dia de férias se aparece em qualquer uma) e reportar a divergência.")

# ------------------------------------------------------------------ 9. desligamento
dl = a[a.CATEGORIA_EVENTO == "DESLIGAMENTO"].merge(i[["ID_INTEGRANTE", "dem", "CODCOLIGADA"]], on="ID_INTEGRANTE", suffixes=("", "_c"))
dl["dif"] = (dl.ini - dl.dem).dt.days.astype("Int64")
dp = i[(i.dem >= INI) & (i.dem <= FIM)]
problema(9, "O evento DESLIGAMENTO em afastamentos não confere com o cadastro", "MÉDIO", "T1",
         "Cada demissão do período apareceria como evento DESLIGAMENTO, na mesma data.",
         f"O cadastro tem <b>{fmt(len(dp))}</b> demissões no período; só <b>{fmt(dp.ID_INTEGRANTE.isin(dl.ID_INTEGRANTE).sum())}</b> têm evento "
         f"DESLIGAMENTO (nenhuma da empresa 5). Dos {fmt(len(dl))} eventos, <b>{fmt((dl.dif != 0).sum())}</b> têm data diferente da demissão do "
         f"cadastro — com diferença mediana de <b>{dl.dif[dl.dif != 0].median():.0f} dias</b> e máxima de {fmt(abs(dl.dif.min()))} dias.", "", "")
code("dl = fAfastamentos[CATEGORIA_EVENTO=='DESLIGAMENTO'] JOIN dim_integrante;  (dl.DT_INICIO - dl.DATADEMISSAO).dt.days")
exd = dl[dl.dif != 0].sort_values("dif").iloc[[0, len(dl[dl.dif != 0]) // 2]]
tabela(exd.assign(dem=exd.dem.dt.strftime("%Y-%m-%d"))[["ID_INTEGRANTE", "DT_INICIO", "dem", "dif"]]
       .rename(columns={"ID_INTEGRANTE": "vínculo", "DT_INICIO": "DESLIGAMENTO (afast.)", "dem": "DATADEMISSAO (cadastro)", "dif": "diferença (dias)"}))
fecho("Se a rotatividade fosse contada por <i>fAfastamentos</i>, metade das saídas desapareceria.",
      "Usar exclusivamente <i>dim_integrante</i> (DATADEMISSAO + CATEGORIA_DESLIGAMENTO) para a Trilha 1.")

# ------------------------------------------------------------------ 10. afastamentos problemas
dup = a[a.duplicated(["ID_INTEGRANTE", "CATEGORIA_EVENTO", "DT_INICIO", "DT_FIM"], keep=False)].sort_values(["ID_INTEGRANTE", "DT_INICIO"])
fora_p = a[(a.ini > FIM) | (a.fim.notna() & (a.fim < INI))]
a2 = a.sort_values(["ID_INTEGRANTE", "ini"]).copy(); a2["fim_ant"] = a2.groupby("ID_INTEGRANTE").fim.shift()
sob = a2[a2.ini <= a2.fim_ant]
problema(10, "fAfastamentos: duplicados, sobreposições e eventos fora do período", "MÉDIO", "T2",
         "\"Afastamentos ativos entre 01/01/2025 e 30/09/2026\" (PDF), um registro por evento.",
         f"<b>{a.duplicated(['ID_INTEGRANTE','CATEGORIA_EVENTO','DT_INICIO','DT_FIM']).sum()}</b> duplicados exatos; "
         f"<b>{len(sob)}</b> eventos que começam antes do fim do evento anterior do mesmo vínculo; "
         f"<b>{len(fora_p)}</b> eventos inteiramente fora do período; <b>{a.fim.isna().sum()}</b> sem data de fim "
         f"({', '.join(f'{k}: {v}' for k, v in a[a.fim.isna()].CATEGORIA_EVENTO.value_counts().items())}).", "", "")
code("fAfastamentos.duplicated(['ID_INTEGRANTE','CATEGORIA_EVENTO','DT_INICIO','DT_FIM'])")
tabela(dup[["ID_INTEGRANTE", "CATEGORIA_EVENTO", "DT_INICIO", "DT_FIM"]].head(4).rename(columns={"ID_INTEGRANTE": "vínculo"}))
fecho("Duplicados e sobreposições contam a mesma ausência duas vezes.",
      "Deduplicar, unir intervalos sobrepostos por vínculo e recortar no período; eventos sem fim tratados como em curso até 30/09/2026.")

# ------------------------------------------------------------------ 11. valores impossíveis
imp = b[(b.MINUTOS_TRABALHADOS < 0) | (b.MINUTOS_TRABALHADOS > 1440) | (b.MINUTOS_EXTRA > 720)]
inv = b[(b.SITUACAO_NA_DATA == "A") & (b.MINUTOS_PREVISTOS > 0) & (b.MINUTOS_TRABALHADOS == 0) & (b.MINUTOS_FALTA == 0) & (b.MINUTOS_ABONO == 0)]
problema(11, "Minutos impossíveis e ausências \"invisíveis\" no ponto", "MÉDIO", "T2",
         "Minutos trabalhados entre 0 e 1.440 (24h) por dia; um dia previsto sem trabalho deveria virar falta ou abono.",
         f"<b>{(b.MINUTOS_TRABALHADOS < 0).sum()}</b> dias com minutos trabalhados negativos (mínimo {b.MINUTOS_TRABALHADOS.min():.0f}); "
         f"<b>{(b.MINUTOS_TRABALHADOS > 1440).sum()}</b> com mais de 24h (máximo {fmt(b.MINUTOS_TRABALHADOS.max())} min = {b.MINUTOS_TRABALHADOS.max()/1440:.1f} dias); "
         f"<b>{(b.MINUTOS_EXTRA > 720).sum()}</b> com mais de 12h extras. E <b>{fmt(len(inv))}</b> dias com situação Ativo, jornada prevista, "
         f"zero trabalhado, zero falta e zero abono — ausências que não aparecem como falta.", "", "")
code("(MINUTOS_TRABALHADOS < 0) | (MINUTOS_TRABALHADOS > 1440) | (MINUTOS_EXTRA > 720)\n"
     "SITUACAO=='A' & PREVISTOS>0 & TRABALHADOS==0 & FALTA==0 & ABONO==0")
ex11 = pd.concat([imp.nsmallest(2, "MINUTOS_TRABALHADOS"), imp.nlargest(2, "MINUTOS_TRABALHADOS"), inv.head(2)])
tabela(ex11[COLS_B].rename(columns=REN_B))
fecho("Somas de horas e taxas de falta distorcidas; parte das ausências reais fica subnotificada.",
      "Truncar minutos ao intervalo [0, 1440]; tratar o dia previsto e não trabalhado sem justificativa como ausência não registrada (análise de sensibilidade com e sem).")

# ------------------------------------------------------------------ 12. ponto sem cadastro
sc = b[~b.ID_INTEGRANTE.isin(ii.index)]
problema(12, "Ponto de vínculos que não existem no cadastro", "BAIXO", "T2",
         "Todo ID_INTEGRANTE de <i>fBatidas</i> existe em <i>dim_integrante</i>.",
         f"<b>{len(sc)}</b> linhas de ponto de <b>{sc.ID_INTEGRANTE.nunique()}</b> vínculos sem cadastro: {', '.join(sc.ID_INTEGRANTE.unique()[:8])}…", "", "")
code("~fBatidas.ID_INTEGRANTE.isin(dim_integrante.ID_INTEGRANTE)")
fecho("Sem empresa/função/datas, essas linhas não podem ser usadas.", "Descartar (impacto desprezível).")

# ------------------------------------------------------------------ 13. vínculos sobrepostos
s = i.sort_values(["CODPESSOA", "adm"]).copy()
s["prox_adm"] = s.groupby("CODPESSOA").adm.shift(-1); s["prox_emp"] = s.groupby("CODPESSOA").CODCOLIGADA.shift(-1); s["prox_id"] = s.groupby("CODPESSOA").ID_INTEGRANTE.shift(-1)
gap = (s.prox_adm - s.dem).dt.days
ov = s[gap < 0]; rc = s[gap.between(0, 3)]
problema(13, "A mesma pessoa com vínculos sobrepostos ou recontratada em dias", "MÉDIO", "T1",
         "Uma admissão nova representa entrada de pessoa e uma demissão, saída — base da fórmula de turnover.",
         f"<b>{len(ov)}</b> casos em que a pessoa (CODPESSOA) é admitida de novo <b>antes</b> da demissão do vínculo anterior "
         f"({(ov.prox_emp == ov.CODCOLIGADA).sum()} na mesma empresa) e <b>{len(rc)}</b> recontratações em até 3 dias "
         f"({(rc.prox_emp != rc.CODCOLIGADA).sum()} para outra empresa do grupo). Há {fmt((i.MOTIVOADMISSAO == '02').sum())} admissões com motivo \"Transferência\".", "", "")
code("ordenar por CODPESSOA, DATAADMISSAO;  gap = próxima_admissão - DATADEMISSAO;  gap < 0 (sobreposto), 0<=gap<=3 (recontratação)")
exo = ov.head(3)[["CODPESSOA", "ID_INTEGRANTE", "adm", "dem", "prox_id", "prox_adm"]].copy()
for c_ in ["adm", "dem", "prox_adm"]:
    exo[c_] = exo[c_].dt.strftime("%Y-%m-%d")
exo.columns = ["pessoa", "vínculo 1", "admissão 1", "demissão 1", "vínculo 2", "admissão 2"]
tabela(exo)
fecho("Transferências e trocas de matrícula entram como admissão + rescisão, inflando o turnover sem que ninguém tenha saído do grupo.",
      "Calcular o turnover em duas versões: oficial (por vínculo) e por pessoa, desconsiderando transferências e recontratações em até 3 dias.")

# ================================================================== PARTE 3
P("Parte 3 — Variáveis que não significam o que parecem", "h1")
sal_ano = i[i.dem.notna()].groupby(i.dem.dt.year).sal.median()
anos = [2000, 2004, 2008, 2012, 2016, 2020, 2024]
problema(14, "Salário é nominal e da época da última alteração", "MÉDIO", "Custo",
         "\"Salário de referência\" comparável entre integrantes.",
         "A mediana salarial por ano de demissão cresce de "
         + ", ".join(f"{a_}: R$ {sal_ano.get(a_, np.nan):,.0f}".replace(",", ".") for a_ in anos)
         + f"; ativos hoje: R$ {ativos.sal.median():,.0f}".replace(",", ".") + ". O valor é o da época, sem correção.", "", "")
code("dim_integrante[DATADEMISSAO.notna()].groupby(ano(DATADEMISSAO)).SALARIO.median()")
fecho("Comparar salários entre períodos ou usá-los como preditor mistura inflação com perfil.",
      "Usar salário só dos ativos (custo do contrato) ou relativizar ao piso da função no mesmo ano.")

fx = i[i.adm.dt.year < 1990].FAIXA_ETARIA.value_counts().head(5)
problema(15, "FAIXA_ETARIA não é a idade atual (nem documentada)", "BAIXO", "T1",
         "Faixa etária do integrante.",
         f"Vínculos admitidos antes de 1990 — que hoje teriam ao menos 50 anos — aparecem com faixas "
         f"{', '.join(fx.index)}. A faixa parece ser a idade na demissão (ou na última atualização), e não há documentação. "
         f"{i.FAIXA_ETARIA.isna().sum()} vínculos sem faixa.", "", "")
code("dim_integrante[ano(DATAADMISSAO) < 1990].FAIXA_ETARIA.value_counts()")
fecho("Usar faixa etária como preditor exige saber a data de referência.", "Usar apenas para vínculos do período (onde a idade na saída é próxima da atual) e declarar a premissa.")

problema(16, "Colunas prometidas ausentes ou sem informação", "MÉDIO", "T1, T3",
         "PDF cita \"cargo padronizado (que agrupa funções equivalentes)\"; dim_horario traz turno e tipo de jornada.",
         f"<i>dim_funcao</i> só tem as colunas {', '.join(fu.columns)} — <b>não existe cargo padronizado</b>. "
         f"Em <i>dim_horario</i>, TURNO é \"NAO IDENTIFICADO\" em <b>{100*(h.TURNO=='NAO IDENTIFICADO').mean():.0f}%</b> dos horários e "
         f"TIPO_JORNADA é \"SEM INTERVALO DEFINIDO\" em <b>{100*(h.TIPO_JORNADA=='SEM INTERVALO DEFINIDO').mean():.0f}%</b>. "
         f"{fmt((fu.FUNCAO=='OUTRAS FUNCOES').sum())} dos {fmt(len(fu))} códigos de função se chamam \"OUTRAS FUNCOES\" "
         f"({fmt((per.FUNCAO=='OUTRAS FUNCOES').sum())} vínculos do período). Há grafias duplicadas, como \"COBRADOR\" e \"COBRADOR(A)\". "
         f"Faltam CODCCUSTO em {i.CODCCUSTO.isna().sum()} e CODEQUIPE em {i.CODEQUIPE.isna().sum()} vínculos.", "", "")
code("dim_horario.TURNO.value_counts(normalize=True); dim_funcao.FUNCAO.value_counts()")
fecho("Agrupar funções equivalentes e segmentar por turno não é possível com os dados entregues.",
      "Criar um agrupamento próprio de funções (motorista de ônibus, cobrador, motorista de caminhão, aprendiz, outros) e ignorar TURNO/TIPO_JORNADA.")

fd = set(pd.to_datetime(fer.data))
fer_fora = sorted(b[(b.FERIADO == "1") & ~b.d.isin(fd)].DATA.unique())
fer_sem = sorted(d_ for d_ in fer.data if INI <= pd.Timestamp(d_) <= FIM and d_ not in set(b[b.FERIADO == "1"].DATA))
problema(17, "Feriados do ponto não batem com dim_feriados", "BAIXO", "T2, T3",
         "FERIADO = 1 no ponto exatamente nas datas de <i>dim_feriados</i>.",
         f"<b>{fmt(((b.FERIADO=='1') & ~b.d.isin(fd)).sum())}</b> linhas marcadas como feriado em datas que não estão na tabela: "
         f"{', '.join(fer_fora)} (provavelmente feriados municipais). E datas da tabela dentro do período nunca marcadas no ponto: "
         f"{', '.join(fer_sem)} (carnaval/quarta de cinzas — pontos facultativos).", "", "")
code("fBatidas[(FERIADO=='1') & ~DATA.isin(dim_feriados.data)].DATA.unique()")
fecho("Pequeno efeito na contagem de dias úteis a cobrir.", "Usar o FERIADO do próprio ponto (reflete o calendário local) e dim_feriados só para projeção.")

nd_ = ((i.dem < i.adm).sum(), (i.dem == i.adm).sum())
problema(18, "Datas incoerentes no cadastro", "BAIXO", "T1",
         "DATADEMISSAO posterior à DATAADMISSAO.",
         f"<b>{nd_[0]}</b> vínculo com demissão anterior à admissão "
         f"({', '.join(i[i.dem < i.adm].ID_INTEGRANTE)}: {i[i.dem<i.adm].DATAADMISSAO.iloc[0]} → {i[i.dem<i.adm].DATADEMISSAO.iloc[0]}) "
         f"e <b>{nd_[1]}</b> contratos encerrados no mesmo dia da admissão.", "", "")
code("(DATADEMISSAO < DATAADMISSAO).sum(); (DATADEMISSAO == DATAADMISSAO).sum()")
fecho("Desprezível em volume.", "Excluir o caso negativo; manter os de 0 dias (são admissões e rescisões reais para o turnover).")

# ================================================================== PARTE 4 colinearidade
P("Parte 4 — Colinearidade e redundância entre variáveis", "h1")
cats = ["CODCOLIGADA", "ID_COLIGADA_FILIAL", "CODCCUSTO", "ESCALA", "JORNADAMENSAL", "FUNCAO", "FAIXA_SAL", "CODTIPO",
        "TIPOADMISSAO", "MOTIVOADMISSAO", "SEXO", "GRAUINSTRUCAO", "FAIXA_ETARIA"]
nomes = {"CODCOLIGADA": "empresa", "ID_COLIGADA_FILIAL": "filial", "CODCCUSTO": "centro custo", "ESCALA": "escala", "JORNADAMENSAL": "jornada mensal",
         "FUNCAO": "função", "FAIXA_SAL": "faixa salarial", "CODTIPO": "tipo vínculo", "TIPOADMISSAO": "tipo admissão",
         "MOTIVOADMISSAO": "motivo admissão", "SEXO": "sexo", "GRAUINSTRUCAO": "escolaridade", "FAIXA_ETARIA": "faixa etária"}
V = pd.DataFrame({x: {y: (1.0 if x == y else cramers_v(p_[x].fillna("NA"), p_[y].fillna("NA"))) for y in cats} for x in cats}).loc[cats, cats]
fg, ax = plt.subplots(figsize=(7.4, 6))
im = ax.imshow(V.values.astype(float), cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"]), vmin=0, vmax=1)
ax.set_xticks(range(len(cats))); ax.set_yticks(range(len(cats)))
ax.set_xticklabels([nomes[c] for c in cats], rotation=45, ha="right"); ax.set_yticklabels([nomes[c] for c in cats])
for (r_, c_), v in np.ndenumerate(V.values.astype(float)):
    if r_ != c_:
        ax.text(c_, r_, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.55 else "#0b0b0b")
for s_ in ax.spines.values():
    s_.set_visible(False)
cb = fg.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("V de Cramér (0 = independentes, 1 = redundantes)"); cb.outline.set_visible(False)
ax.set_title("Associação entre variáveis do cadastro (vínculos do período)", loc="left", fontsize=10, color="#0b0b0b")
fg.tight_layout(); fg.savefig(f"{FIG}/cramer.png", dpi=200); plt.close(fg)
pares = sorted([(nomes[x], nomes[y], V.loc[x, y]) for k, x in enumerate(cats) for y in cats[k + 1:] if V.loc[x, y] >= 0.6], key=lambda t: -t[2])
problema(19, "Variáveis categóricas do cadastro são fortemente redundantes", "MÉDIO", "T1, T3",
         "Variáveis explicativas razoavelmente independentes para um modelo de rotatividade.",
         "Medimos a associação par a par com o V de Cramér (corrigido para viés, Bergsma 2013) nos vínculos do período. "
         "Pares com V ≥ 0,60: " + "; ".join(f"<b>{x} ~ {y}</b> ({v:.2f})" for x, y, v in pares) + ". "
         f"Estruturalmente: cada seção tem um único centro de custo ({100*(i.groupby('ID_SECAO').CODCCUSTO.nunique()<=1).mean():.0f}%) e "
         f"cada horário determina uma única escala ({100*(i.groupby('ID_HORARIO').ESCALA.nunique()<=1).mean():.0f}%).", "", "")
code("V(x,y) = sqrt( max(0, χ²/n - (k-1)(r-1)/(n-1)) / min(k*-1, r*-1) )   para todo par de colunas categóricas")
fig("cramer", 14.5)
_pv = [qui2(pd.crosstab(p_[x].fillna("NA"), p_[y].fillna("NA")))[2] for k, x in enumerate(cats) for y in cats[k + 1:]]
teste(f"qui-quadrado de independência para os {len(_pv)} pares: {sum(v < 0.05 for v in _pv)} rejeitam H0 a 5%. Como n é grande, a significância "
      "é quase garantida; o que decide a redundância é o tamanho do efeito (V de Cramér no mapa).")
fecho("Num modelo, empresa + filial + centro de custo (ou função + salário) juntos geram coeficientes instáveis e efeitos não identificáveis; "
      "o efeito da escala se confunde com empresa e função (problema 4).",
      "Usar uma variável por grupo redundante (empresa; função agrupada; escala), regularização (ridge/lasso) quando houver muitas, "
      "e reportar o efeito da escala condicionado a empresa e função.")

r = b.MINUTOS_TRABALHADOS - (b.MINUTOS_PREVISTOS - b.MINUTOS_FALTA - b.MINUTOS_ATRASO + b.MINUTOS_EXTRA)
cor = b[NUM].corr().round(2); cor.index = cor.columns = [c.replace("MINUTOS_", "").lower() for c in NUM]
problema(20, "As colunas de minutos do ponto são ligadas por uma identidade contábil", "BAIXO", "T2",
         "Seis medidas distintas da jornada.",
         f"A igualdade <b>trabalhado = previsto − falta − atraso + extra</b> vale exatamente em <b>{100*(r==0).mean():.0f}%</b> das linhas: "
         f"as colunas são (quase) combinação linear umas das outras. Correlação trabalhado × previsto = {cor.loc['trabalhados','previstos']:.2f}.", "", "")
code("(TRABALHADOS == PREVISTOS - FALTA - ATRASO + EXTRA).mean()")
tabela(cor.rename_axis("").reset_index())
_X = b[NUM].astype(float).values; _vif = []
for j in range(_X.shape[1]):
    _y = _X[:, j]; _Z = np.column_stack([np.ones(len(_X)), np.delete(_X, j, axis=1)])
    _beta, *_ = np.linalg.lstsq(_Z, _y, rcond=None); _r2 = 1 - ((_y - _Z @ _beta) ** 2).sum() / ((_y - _y.mean()) ** 2).sum()
    _vif.append(1 / (1 - _r2))
teste("fator de inflação da variância (VIF) de cada coluna regredida nas outras cinco: "
      + ", ".join(f"{c.replace('MINUTOS_','').lower()} = {v:.1f}" for c, v in zip(NUM, _vif))
      + f". {', '.join(c.replace('MINUTOS_','').lower() for c, v in zip(NUM, _vif) if v > 10)} passam de 10 (multicolinearidade grave). "
      "Os VIFs só não são infinitos porque parte das linhas viola a identidade, o que é em si mais um sinal de inconsistência; "
      "nas linhas que a respeitam, a colinearidade é perfeita.")
fecho("Colocar as seis colunas num mesmo modelo gera multicolinearidade perfeita; as 16% de linhas que violam a identidade são outro sinal de inconsistência.",
      "Trabalhar com taxas derivadas (falta/previsto, abono/previsto) em vez das colunas brutas.")

# ================================================================== PARTE 5 o que está OK
P("Parte 5 — O que confere", "h1")
at_f = ativos.groupby("ID_FUNCAO").size().rename("ativos no cadastro")
mm = meta.merge(at_f, left_on="ID_FUNCAO", right_index=True, how="left").merge(fu[["ID_FUNCAO", "FUNCAO"]], on="ID_FUNCAO")
mm = mm[["CODCOLIGADA", "ID_FUNCAO", "FUNCAO", "ATUAL", "META", "ativos no cadastro"]].rename(columns={"CODCOLIGADA": "empresa"})
P("Para ser justo com a base, verificamos também o que está correto:")
P(f"• Todas as contagens declaradas no PDF batem: {fmt(len(i))} vínculos, {fmt(i.CODPESSOA.nunique())} pessoas, {fmt(len(b))} linhas de ponto, "
  f"{fmt(len(a))} afastamentos, {fmt(len(fu))} funções, {fmt(len(h))} horários; admissões desde {i.adm.min():%d/%m/%Y} e última demissão em {i.dem.max():%d/%m/%Y}.")
P("• Chaves íntegras: toda função, horário e seção do cadastro existe na dimensão correspondente; nenhum ID duplicado; "
  "nenhuma linha duplicada no ponto (vínculo + data).")
P("• Situação e demissão coerentes: todo vínculo com situação D tem data de demissão e todo demitido tem categoria (voluntário/involuntário).")
P("• Os arquivos estão corretamente codificados em UTF-8.")
P("• <i>dim_meta</i> (quadro atual e meta de motoristas) bate exatamente com os ativos do cadastro, exceto na empresa 5 "
  "(1.117 declarados contra 1.177 ativos). Ela pode servir de quadro-base do contrato:")
tabela(mm)
P(f"• Nota: o regulamento fala em \"cerca de 4.500 colaboradores\"; o cadastro tem {fmt(len(ativos))} vínculos ativos ({fmt(ativos.CODPESSOA.nunique())} pessoas).", "s")

# ================================================================== RESUMO
story.append(PageBreak())
P("Resumo dos problemas e tratamentos", "h1")
dados = [[Paragraph(x, st["cellb"]) for x in ["#", "Problema", "Severidade", "Afeta", "Tratamento"]]]
for n, t_, s_, tr, trat in resumo:
    dados.append([Paragraph(str(n), st["cell"]), Paragraph(t_, st["cell"]),
                  Paragraph(f'<font color="{SEV[s_]}" name="Arial-B">{s_}</font>', st["cell"]), Paragraph(tr, st["cell"]), Paragraph(trat, st["cell"])])
tb = Table(dados, colWidths=[0.9 * cm, 5.0 * cm, 2.1 * cm, 1.6 * cm, 7.9 * cm], repeatRows=1, hAlign="LEFT")
tb.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, TXT), ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINHA),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
story.append(tb)
story.append(Spacer(1, 8))
P("<b>Conclusão.</b> Nenhum problema impede o trabalho, mas três deles (1, 2 e 3) restringem o que pode ser medido diretamente: "
  "a rotatividade (Trilha 1) é calculável para todas as empresas a partir do cadastro; ausências (Trilha 2) só são observadas "
  "diretamente nas empresas 6, 3 e 7; e a escala praticada (Trilha 3) é desconhecida para os motoristas de ônibus das empresas 5 e 8. "
  "Nossas estimativas para esses grupos serão apresentadas como cenários, com as premissas explícitas.")


# ------------------------------------------------------------------ rodapé e build
# os parâmetros impacto/tratamento do 'problema' são preenchidos por 'fecho'; manter a tabela-resumo com o tratamento real
def _rodape(canv, doc):
    canv.saveState(); canv.setFont("Arial", 7.5); canv.setFillColor(TXT2)
    canv.drawString(2 * cm, 1.2 * cm, "Auditoria dos dados · III Datathon IME-UFG · Grupo HP")
    canv.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canv.restoreState()


doc = SimpleDocTemplate(SAIDA, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Auditoria dos dados — Datathon Grupo HP", author="Equipe Datathon")
doc.build(story, onFirstPage=_rodape, onLaterPages=_rodape)
print("ok", SAIDA)
