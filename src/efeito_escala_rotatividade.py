"""Trilha 3: associação entre a escala praticada (6x1 x 5x2) e a taxa de desligamento.

Poisson para a taxa mensal de saída por vínculo (offset: log dos meses de exposição; erros robustos),
com a escala medida no ponto e controles para empresa, grupo de função e tempo de casa < 1 ano.

Uso: python src/efeito_escala_rotatividade.py dados
"""
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
DADOS = sys.argv[1] if len(sys.argv) > 1 else "dados"
CORTE = pd.Timestamp("2026-09-30")
rd = lambda n, **k: pd.read_csv(f"{DADOS}/{n}.csv", sep=";", encoding="utf-8-sig", **k)

b = rd("fBatidas", usecols=["ID_INTEGRANTE", "CODCOLIGADA", "DATA", "SITUACAO_NA_DATA", "MINUTOS_PREVISTOS", "FERIADO"], parse_dates=["DATA"])
i = rd("dim_integrante", usecols=["ID_INTEGRANTE", "ID_FUNCAO", "DATAADMISSAO", "DATADEMISSAO", "CATEGORIA_DESLIGAMENTO"], parse_dates=["DATAADMISSAO", "DATADEMISSAO"])
f = rd("dim_funcao", usecols=["ID_FUNCAO", "FUNCAO"])

x = b.merge(i, on="ID_INTEGRANTE").merge(f, on="ID_FUNCAO")
x = x[(x.DATA >= x.DATAADMISSAO) & (x.DATADEMISSAO.isna() | (x.DATA <= x.DATADEMISSAO))]   # só dias do contrato
x = x[~x.FUNCAO.str.contains("APRENDIZ", na=False)]

# escala praticada: dias com jornada prevista por semana, nos dias normais (ativo, sem feriado)
n = x[(x.SITUACAO_NA_DATA == "A") & (x.FERIADO == 0)]
g = n.groupby("ID_INTEGRANTE").agg(dias=("DATA", "size"), trab=("MINUTOS_PREVISTOS", lambda s: (s > 0).sum()))
g = g[g.dias >= 28]
g["dsem"] = 7 * g.trab / g.dias
g["escala"] = np.select([g.dsem >= 5.5, (g.dsem >= 4.5) & (g.dsem < 5.5)], ["6x1", "5x2"], "outra")

e = x.groupby("ID_INTEGRANTE").agg(ini=("DATA", "min"), fim=("DATA", "max"), emp=("CODCOLIGADA", "first"), fun=("FUNCAO", "first"),
                                     dem=("DATADEMISSAO", "first"), cat=("CATEGORIA_DESLIGAMENTO", "first"), adm=("DATAADMISSAO", "first"))
d = g.join(e, how="inner")
d = d[d.escala != "outra"].copy()
d["expo"] = ((d.fim - d.ini).dt.days + 1) / 30.44
d["saiu"] = (d.dem.notna() & (d.dem <= CORTE)).astype(int)
d["vol"] = ((d.saiu == 1) & (d.cat == "VOLUNTARIO")).astype(int)
d["grupo"] = np.where(d.fun.str.contains("MOTORISTA", na=False) & ~d.fun.str.contains("ONIBUS", na=False), "mot_caminhao", "demais")
d["novato"] = ((d.ini - d.adm).dt.days < 365).astype(int)
d = d[d.expo > 0]

print("vínculos:", len(d))
print(d.groupby("escala").agg(n=("saiu", "size"), saidas=("saiu", "sum"), voluntarias=("vol", "sum"), meses=("expo", "sum")).round(1))
for alvo in ("saiu", "vol"):
    for nome, sub in (("todas as empresas", d), ("só empresa 6", d[d.emp == 6])):
        X = sm.add_constant(pd.get_dummies(sub[["escala", "grupo", "emp", "novato"]].astype(str), drop_first=True).astype(float))
        m = sm.GLM(sub[alvo], X, family=sm.families.Poisson(), offset=np.log(sub.expo)).fit(cov_type="HC0")
        k = [c for c in X.columns if c.startswith("escala")][0]
        lo, hi = np.exp(m.conf_int().loc[k])
        print(f"{alvo:5s} | {nome:18s} | RR 6x1/5x2 = {np.exp(m.params[k]):.2f}  IC95% [{lo:.2f}, {hi:.2f}]  p = {m.pvalues[k]:.3f}  n = {len(sub)}")
