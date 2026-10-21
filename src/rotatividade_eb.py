"""Trilha 1: encolhimento bayesiano empírico entre empresas e faixas calibradas fora da amostra.

Modelo gama-Poisson para a taxa de cada empresa, r_e = y_e / E_e, com y = (admissões + rescisões) / 2 e
E = soma do headcount mensal. Estimativa encolhida: (y_e + m b) / (E_e + b), com m a taxa do grupo e
b = m / tau2, tau2 a variância entre empresas (momentos).
"""
import numpy as np
import pandas as pd
import rotatividade as rot


def taxas_eb(S, origem, meses=36):
    """Turnover anual (%) bruto e encolhido de cada empresa, com dados até `origem`."""
    lin = []
    for e, s in S.items():
        w = s[s.mes <= origem].tail(meses)
        if w.hc.sum() > 0:
            lin.append((e, (w.adm + w.resc).sum() / 2, w.hc.sum()))
    t = pd.DataFrame(lin, columns=['empresa', 'y', 'E']).set_index('empresa')
    r = t.y / t.E
    m = t.y.sum() / t.E.sum()
    tau2 = max(np.average((r - m) ** 2, weights=t.E) - m * len(t) / t.E.sum(), 1e-12)
    b = m / tau2
    return ((t.y + m * b) / (t.E + b) * 1200).rename('eb'), (r * 1200).rename('bruta'), b


def backtest(S, H, inicio='2019-12-01', fim='2026-09-01', empresas=(3, 5, 6, 7, 8)):
    """Projeção (média 36m e encolhida) x turnover realizado nos H meses seguintes, por origem."""
    lin = []
    for o in pd.date_range(inicio, pd.Timestamp(fim) - pd.DateOffset(months=H), freq='MS'):
        eb, _, _ = taxas_eb(S, o)
        for e in empresas:
            h = S[e][S[e].mes <= o]
            f = S[e][(S[e].mes > o) & (S[e].mes <= o + pd.DateOffset(months=H))]
            if len(f) < H or h.hc.tail(12).mean() < 20 or e not in eb.index:
                continue
            lin.append({'empresa': e, 'origem': o, 'H': H, 'real': rot.turnover_janela(f) * 12 / H,
                        'media36': rot.taxa_anual(h, 36), 'eb36': eb[e]})
    return pd.DataFrame(lin)


def faixa_lognormal(razoes_empresa, razoes_grupo, z=1.2816, n0=12):
    """Faixa da razão realizado/projetado em escala log, com centro e dispersão da empresa encolhidos
    para os do grupo (peso n/(n + n0)). z = 1,2816 corresponde a P10-P90."""
    la = np.log(np.clip(razoes_empresa, .05, 20))   # razão 0 (sem movimento) não pode ir ao log
    lg = np.log(np.clip(razoes_grupo, .05, 20))
    w = len(la) / (len(la) + n0)
    centro = w * (la.median() if len(la) else 0) + (1 - w) * lg.median()
    sd = np.sqrt(w * (la.var() if len(la) > 1 else 0) + (1 - w) * lg.var())
    return np.exp(centro - z * sd), np.exp(centro), np.exp(centro + z * sd)


def _passado(bt, r):
    return bt[bt.origem + pd.DateOffset(months=int(r.H)) <= r.origem]


def cobertura_lognormal(bt, metodo='media36', z=1.2816, min_hist=24):
    """Cobertura fora da amostra da faixa log-normal: em cada origem, só usa horizontes já encerrados."""
    bt = bt.assign(razao=bt.real / bt[metodo])
    saida = []
    for r in bt.itertuples():
        p = _passado(bt, r)
        if len(p) >= min_hist:
            lo, _, hi = faixa_lognormal(p[p.empresa == r.empresa].razao, p.razao, z)
            saida.append({'empresa': r.empresa, 'fora': lo <= r.razao <= hi})
    return pd.DataFrame(saida)


def cobertura_faixa(bt, metodo='media36', q=(.1, .9), min_hist=12):
    """Cobertura da faixa por quantis de cada empresa (método do notebook 02), calibrada com todas as
    origens ('dentro') ou só com horizontes já encerrados ('fora')."""
    bt = bt.assign(razao=bt.real / bt[metodo])
    saida = []
    for e, g in bt.groupby('empresa'):
        lo_in, hi_in = g.razao.quantile(q[0]), g.razao.quantile(q[1])
        for r in g.itertuples():
            p = _passado(g, r)
            if len(p) >= min_hist:
                lo, hi = p.razao.quantile(q[0]), p.razao.quantile(q[1])
                saida.append({'empresa': e, 'dentro': lo_in <= r.razao <= hi_in, 'fora': lo <= r.razao <= hi})
    return pd.DataFrame(saida)
