"""Funções da Trilha 1 (rotatividade).

Turnover (%) = [(Admissões + Rescisões) / 2] / Headcount × 100

Unidade de análise: empresa (CODCOLIGADA). Headcount do mês = vínculos que
existiram em algum momento do mês (admitidos até o fim e não demitidos antes do início).
"""
import numpy as np
import pandas as pd

CORTE = pd.Timestamp('2026-09-30')  # último dia com dados de ponto/afastamento
FAIXAS_TEMPO_CASA = [-1, 0, 1, 2, 5, 11, 23, 59, 10**6]  # meses completos de casa
ROTULOS_FAIXAS = ['0', '1', '2', '3-5', '6-11', '12-23', '24-59', '60+']


def carregar_integrantes(caminho):
    cols = ['ID_INTEGRANTE', 'CODCOLIGADA', 'DATAADMISSAO', 'DATADEMISSAO', 'CATEGORIA_DESLIGAMENTO']
    return pd.read_csv(caminho, sep=';', encoding='utf-8-sig', usecols=cols,
                       parse_dates=['DATAADMISSAO', 'DATADEMISSAO'])


def serie_mensal(integrantes, inicio='2016-01-01', fim='2026-09-01'):
    """Admissões, rescisões e headcount por empresa e mês (índice: empresa, mes)."""
    meses = pd.date_range(inicio, fim, freq='MS')
    saida = []
    for emp, g in integrantes.groupby('CODCOLIGADA'):
        adm = g.DATAADMISSAO.dt.to_period('M').dt.to_timestamp().value_counts()
        resc = g.DATADEMISSAO.dropna().dt.to_period('M').dt.to_timestamp().value_counts()
        adm = adm.reindex(meses, fill_value=0)
        resc = resc.reindex(meses, fill_value=0)
        # vínculos existentes durante o mês = admitidos até o mês - demitidos antes do mês
        antes = g.DATADEMISSAO.dropna().dt.to_period('M').dt.to_timestamp()
        acum_adm = g.DATAADMISSAO.dt.to_period('M').dt.to_timestamp().value_counts().sort_index().cumsum()
        acum_resc_antes = antes.value_counts().sort_index().cumsum()
        hc = (acum_adm.reindex(meses, method='ffill').fillna(0)
              - acum_resc_antes.reindex(meses - pd.DateOffset(months=1), method='ffill').fillna(0).values)
        saida.append(pd.DataFrame({'empresa': emp, 'mes': meses, 'adm': adm.values,
                                   'resc': resc.values, 'hc': hc.values}))
    return pd.concat(saida, ignore_index=True)


def turnover_janela(janela):
    """Turnover (%) da janela pela fórmula do enunciado: soma de adm e resc / 2 / headcount médio."""
    hc = janela.hc.mean()
    return np.nan if not hc else (janela.adm.sum() + janela.resc.sum()) / 2 / hc * 100


def taxa_anual(historico, meses):
    """Turnover mensal anualizado: taxa mensal agregada dos últimos `meses` × 12.

    Usa o headcount de cada mês. Não é a fórmula do enunciado aplicada a um ano inteiro (ver
    `turnover_periodo`), que usa como denominador todos os vínculos do ano e dá valores menores.
    """
    w = historico.tail(meses)
    return (w.adm + w.resc).sum() / 2 / w.hc.sum() * 100 * 12


def painel_tempo_casa(integrantes, inicio='2018-01-01', fim='2026-09-01'):
    """Painel vínculo-mês: ativo no início do mês, tempo de casa e se saiu no mês."""
    adm = integrantes.DATAADMISSAO.values.astype('datetime64[D]')
    dem = integrantes.DATADEMISSAO.values.astype('datetime64[D]')
    emp = integrantes.CODCOLIGADA.values
    partes = []
    for ini in pd.date_range(inicio, fim, freq='MS'):
        a = np.datetime64(ini, 'D')
        b = np.datetime64(ini + pd.DateOffset(months=1), 'D')
        ativo = (adm < a) & (np.isnat(dem) | (dem >= a))
        saiu = ativo & ~np.isnat(dem) & (dem < b)
        tempo = ((a - adm[ativo]).astype(int) / 30.44).astype(int)
        partes.append(pd.DataFrame({'mes': ini, 'emp': emp[ativo], 'tempo': tempo, 'saiu': saiu[ativo].astype(int)}))
    p = pd.concat(partes, ignore_index=True)
    p['faixa'] = pd.cut(p.tempo, FAIXAS_TEMPO_CASA, labels=False)
    return p


def risco_por_faixa(painel, empresa, origem, janela=36, peso_prior=30):
    """Risco mensal de saída por faixa de tempo de casa, com encolhimento para o risco do grupo."""
    n = len(ROTULOS_FAIXAS)
    w = painel[(painel.mes >= origem - pd.DateOffset(months=janela - 1)) & (painel.mes <= origem)]
    emp = w[w.emp == empresa].groupby('faixa').saiu.agg(['sum', 'count']).reindex(range(n)).fillna(0)
    grp = w.groupby('faixa').saiu.agg(['sum', 'count']).reindex(range(n)).fillna(0)
    prior = (grp['sum'] + 0.5) / (grp['count'] + 50)
    return ((emp['sum'] + peso_prior * prior) / (emp['count'] + peso_prior)).values


def simular_coorte(integrantes, historico, empresa, origem, risco, meses=12, adm_mensal=None):
    """Projeta adm/resc/hc mês a mês envelhecendo o quadro atual com o risco por tempo de casa.

    adm_mensal=None repõe exatamente as saídas (quadro estável); um número fixa as admissões mensais.
    """
    inicio_prox = origem + pd.DateOffset(months=1)
    e = integrantes[(integrantes.CODCOLIGADA == empresa) & (integrantes.DATAADMISSAO < inicio_prox)
                    & (integrantes.DATADEMISSAO.isna() | (integrantes.DATADEMISSAO >= inicio_prox))]
    tempo = ((inicio_prox - e.DATAADMISSAO).dt.days / 30.44).astype(int).clip(upper=199).values
    n = np.bincount(tempo, minlength=200).astype(float)
    faixa = np.searchsorted(FAIXAS_TEMPO_CASA[1:], np.arange(200), side='left')
    h = risco[faixa]
    saida = []
    for _ in range(meses):
        sai = n * h
        adm = sai.sum() if adm_mensal is None else adm_mensal
        saida.append((adm, sai.sum(), n.sum() + adm))
        n = envelhecer(n - sai, adm)
    return pd.DataFrame(saida, columns=['adm', 'resc', 'hc'])


def envelhecer(n, entradas):
    """Avança um mês de casa: entradas vão para o compartimento 0; o último (199+ meses) acumula quem chega a ele."""
    novo = np.concatenate([[entradas], n[:-1]])
    novo[-1] += n[-1]
    return novo


def turnover_periodo(integrantes, empresa, inicio, fim):
    """Turnover (%) pela fórmula do enunciado aplicada ao período inteiro [inicio, fim].

    Headcount = vínculos com algum dia de vínculo no período. Para um ano, o denominador inclui todos os
    que passaram pela empresa no ano, e o valor fica abaixo da taxa mensal anualizada (`taxa_anual`).
    """
    ini, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
    g = integrantes[integrantes.CODCOLIGADA == empresa]
    adm = g.DATAADMISSAO.between(ini, fim).sum()
    resc = g.DATADEMISSAO.between(ini, fim).sum()
    hc = ((g.DATAADMISSAO <= fim) & (g.DATADEMISSAO.isna() | (g.DATADEMISSAO >= ini))).sum()
    return np.nan if not hc else (adm + resc) / 2 / hc * 100
