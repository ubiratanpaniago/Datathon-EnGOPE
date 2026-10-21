# -*- coding: utf-8 -*-
"""
III Datathon IME-UFG (VII ENGOPE) — parceiro SIAN / Grupo HP
Solução completa em um único arquivo: Trilha 1 (rotatividade), Trilha 2 (folguistas e feristas),
Trilha 3 (escala 6x1 para 5x2) e modelo integrado com impacto em R$.

COMO EXECUTAR
    pip install pandas numpy scipy statsmodels      (Python 3.10 ou superior; matplotlib é opcional)
    python solucao_datathon.py

    Os CSVs fornecidos pela organização devem estar na pasta `dados/`, ao lado deste arquivo.
    Os resultados são gravados em `resultados/` (criada automaticamente). Tempo de execução: menos de 1 minuto.

    GRÁFICOS (opcional): com `pip install matplotlib`, os gráficos são salvos como PNG em `figuras/`.
    Sem o matplotlib, o script roda normalmente e só pula os gráficos.

ORGANIZAÇÃO DO ARQUIVO
    Parte A. Funções reutilizáveis (rotatividade, ausências, escala, modelo integrado) e auditoria dos dados.
    Parte B. Trilha 1: série mensal, diagnóstico, backtest de estratégias e projeção.
    Parte C. Trilha 2: classificação e limpeza do ponto, taxas por causa, validação, reserva e quadro extra.
    Parte D. Trilha 3: jornada praticada por escala, cenários e quadro adicional.
    Parte E. Modelo integrado: as três trilhas numa base única (a posição) e o custo em reais.
    Parte F. Extensões: faixas da T1 validadas fora da amostra, estimador bayesiano empírico, reserva da T2
             escolhida por custo esperado e efeito da escala sobre a rotatividade (modelo de Poisson).

    O texto explicativo de cada etapa aparece como comentário antes do código correspondente.
    A mesma análise, com gráficos, está nos notebooks do repositório da equipe (os comentários citam
    "notebook 02" a "notebook 06": correspondem às Partes B a F deste arquivo).
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import binom              # reserva por custo esperado (Parte F)
import statsmodels.api as sm               # modelo de Poisson da escala x rotatividade (Parte F)
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # executa a partir da pasta do script (dados/ e resultados/ ao lado)
os.makedirs('resultados', exist_ok=True)

try:                                                     # gráficos são opcionais: o cálculo não depende deles
    import matplotlib
    matplotlib.use('Agg')                                # salva em arquivo, sem abrir janelas
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 100, 'axes.grid': True, 'grid.alpha': .3})
    os.makedirs('figuras', exist_ok=True)
    TEM_MPL = True
except ImportError:
    TEM_MPL = False
    print('matplotlib não instalado: os gráficos serão pulados (os resultados não mudam).')
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 100)


# ====================================================================================================
# PARTE A. FUNÇÕES REUTILIZÁVEIS
# ====================================================================================================

# ---------------------------------------------------------------- funções: rotatividade
"""Funções da Trilha 1 (rotatividade).

Turnover (%) = [(Admissões + Rescisões) / 2] / Headcount × 100

Unidade de análise: empresa (CODCOLIGADA). Headcount do mês = vínculos que
existiram em algum momento do mês (admitidos até o fim e não demitidos antes do início).
"""

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

# ---------------------------------------------------------------- funções: ausencias
"""Funções da Trilha 2 (folguistas e feristas).

Ideia central: a "perda de carga" é a fração do trabalho programado que deixa de ser
entregue por ausência. Para entregar N posições, a folha precisa de N / (1 - perda) pessoas,
logo o quadro extra é N * perda / (1 - perda).

A perda é separada em três blocos, que se tratam de formas diferentes:
  ferias      planejada; só quem tem >= 1 ano de casa tira férias
  prolongada  INSS, acidente, licença, aposentadoria por incapacidade (estado persistente)
  curta       faltas, atestados de até 15 dias e abonos (imprevista, varia dia a dia)
"""

# situação do ponto -> bloco de ausência (F férias, P doença/INSS, T acidente, I/L/M/E/Y licenças e outros)
SITUACAO_BLOCO = {'F': 'ferias', 'T': 'prolongada', 'I': 'prolongada', 'L': 'prolongada',
                  'M': 'prolongada', 'E': 'prolongada', 'Y': 'prolongada'}
BLOCOS = ['ferias', 'prolongada', 'curta']
CORTE = pd.Timestamp('2026-09-30')


def grupo_funcao(nome):
    """Agrupa as funções (dim_funcao) em grupos operacionais."""
    n = str(nome).upper()
    if 'APRENDIZ' in n:
        return 'Aprendiz'
    if 'MOTORISTA' in n and 'ONIBUS' in n:
        return 'Motorista de ônibus'
    if 'MOTORISTA' in n:
        return 'Motorista (caminhão e leves)'
    if 'COBRADOR' in n:
        return 'Cobrador'
    return 'Demais'


def _eventos_por_dia(afast, categorias):
    ev = afast[afast.CATEGORIA_EVENTO.isin(categorias)].copy()
    ev['DT_FIM'] = ev.DT_FIM.fillna(CORTE)
    ev = ev[ev.DT_FIM >= '2025-01-01']
    ev['DT_INICIO'] = ev.DT_INICIO.clip(lower=pd.Timestamp('2025-01-01'))
    partes = [pd.DataFrame({'ID_INTEGRANTE': r.ID_INTEGRANTE, 'DATA': pd.date_range(r.DT_INICIO, r.DT_FIM)})
              for r in ev.itertuples()]
    return pd.concat(partes).drop_duplicates() if partes else pd.DataFrame(columns=['ID_INTEGRANTE', 'DATA'])


def carregar_ponto(dados):
    """Ponto diário classificado: minutos esperados e minutos perdidos por bloco de ausência."""
    rd = lambda n, **k: pd.read_csv(f'{dados}/{n}.csv', sep=';', encoding='utf-8-sig', **k)
    b = rd('fBatidas', parse_dates=['DATA'])
    i = rd('dim_integrante', usecols=['ID_INTEGRANTE', 'ID_FUNCAO', 'DATAADMISSAO'], parse_dates=['DATAADMISSAO'])
    f = rd('dim_funcao', usecols=['ID_FUNCAO', 'FUNCAO'])
    h = rd('dim_horario', usecols=['ID_HORARIO', 'ESCALA'])
    a = rd('fAfastamentos', parse_dates=['DT_INICIO', 'DT_FIM'])

    d = (b.merge(i, on='ID_INTEGRANTE', how='left').merge(f, on='ID_FUNCAO', how='left')
          .merge(h, on='ID_HORARIO', how='left'))
    d = d.rename(columns={'CODCOLIGADA': 'empresa', 'ESCALA': 'escala'})
    d['situacao'] = d.SITUACAO_NA_DATA.fillna('?')
    d = d[d.situacao != 'D'].copy()                     # demitido não faz parte do quadro
    d['grupo'] = d.FUNCAO.map(grupo_funcao)
    d['anos_casa'] = (d.DATA - d.DATAADMISSAO).dt.days / 365.25

    # minutos esperados num dia de ausência sem previsto = média diária (folgas incluídas) do próprio integrante
    normal = d[(d.situacao == 'A') & (d.FERIADO == 0)]
    e0 = normal.groupby('ID_INTEGRANTE').MINUTOS_PREVISTOS.mean().rename('e0')
    d = d.merge(e0, left_on='ID_INTEGRANTE', right_index=True, how='left')
    d['e0'] = d.e0.fillna(normal.MINUTOS_PREVISTOS.mean())

    # doença de até 15 dias x INSS (ambos aparecem como situação P no ponto)
    inss = _eventos_por_dia(a, ['INSS']).assign(inss=True)
    d = d.merge(inss, on=['ID_INTEGRANTE', 'DATA'], how='left')
    d['inss'] = d.inss.fillna(False).astype(bool)

    bloco = d.situacao.map(SITUACAO_BLOCO)
    bloco = bloco.where(d.situacao != 'P', np.where(d.inss, 'prolongada', 'curta'))
    d['bloco_dia'] = bloco                              # bloco da situação do dia (NaN = dia normal)
    ausente = d.bloco_dia.notna()

    previsto = d.MINUTOS_PREVISTOS
    perda_registrada = np.minimum(previsto, d.MINUTOS_FALTA + d.MINUTOS_ABONO)
    d['esperado'] = np.where(ausente & (previsto == 0), d.e0, previsto)
    perda_dia = np.where(previsto == 0, d.e0, perda_registrada)   # dia de ausência sem previsto perde a carga esperada
    for blk in BLOCOS:
        d['perda_' + blk] = np.where(d.bloco_dia == blk, perda_dia, 0.0)
    # dias normais: faltas e abonos viram ausência curta
    d.loc[~ausente, 'perda_curta'] = perda_registrada[~ausente]
    d['perda_total'] = d[['perda_' + k for k in BLOCOS]].sum(axis=1)
    d['elegivel_ferias'] = d.anos_casa >= 1
    return d.drop(columns=['SITUACAO_NA_DATA'])


def limpar_ponto(d, limiar_falta=0.5, limiar_abono=0.15, ate='2025-04-01'):
    """Remove dois problemas de dados do ponto e devolve (ponto limpo, relatório).

    1. Início sem registro: nos primeiros meses de 2025, empresas cujo ponto ainda não registrava batidas
       aparecem com ~100% de falta. Para cada empresa, descarta-se tudo até o último dia em que mais da
       metade dos programados aparece como falta total (só dias antes de `ate`).
    2. Abono coletivo (Carnaval, 24 e 31/12): dias em que mais de `limiar_abono` da carga de toda a
       população operacional recebeu abono. São previsíveis pelo calendário e não são imprevisto.
    """
    prog = d[(d.MINUTOS_PREVISTOS > 0) & d.bloco_dia.isna() & (d.FERIADO == 0)]
    total = (prog.MINUTOS_FALTA >= 0.9 * prog.MINUTOS_PREVISTOS).groupby([prog.empresa, prog.DATA]).agg(['mean', 'size'])
    marcados = total[(total['mean'] > limiar_falta) & (total['size'] >= 5)].reset_index()
    marcados = marcados[marcados.DATA < pd.Timestamp(ate)]
    corte_empresa = marcados.groupby('empresa').DATA.max()
    sem_registro = d.empresa.map(corte_empresa).notna() & (d.DATA <= d.empresa.map(corte_empresa))

    op = d[(d.grupo != 'Aprendiz') & ~sem_registro]
    ab = np.where(op.bloco_dia.isna(), np.minimum(op.MINUTOS_PREVISTOS - np.minimum(op.MINUTOS_PREVISTOS, op.MINUTOS_FALTA), op.MINUTOS_ABONO), 0)
    dia = op.assign(ab=ab).groupby('DATA').agg(ab=('ab', 'sum'), e=('esperado', 'sum'), n=('ID_INTEGRANTE', 'nunique'))
    coletivos = dia[(dia.n >= 100) & (dia.ab / dia.e > limiar_abono)].index
    sem_registro_ou_coletivo = sem_registro | d.DATA.isin(coletivos)
    rel = {'corte_inicio_por_empresa': corte_empresa.dt.strftime('%Y-%m-%d').to_dict(),
           'linhas_removidas_sem_registro': int(sem_registro.sum()),
           'dias_abono_coletivo': [x.strftime('%Y-%m-%d') for x in coletivos],
           'linhas_removidas_abono_coletivo': int((d.DATA.isin(coletivos) & ~sem_registro).sum()),
           'linhas_antes': len(d), 'linhas_depois': int((~sem_registro_ou_coletivo).sum())}
    return d[~sem_registro_ou_coletivo].copy(), rel


def agregar_por_integrante(d, chaves=('empresa', 'grupo')):
    """Somas por integrante (base para taxas e para o bootstrap por integrante)."""
    d = d.assign(esp_1a=np.where(d.elegivel_ferias, d.esperado, 0.0),
                 ferias_1a=np.where(d.elegivel_ferias, d.perda_ferias, 0.0))
    return (d.groupby(list(chaves) + ['ID_INTEGRANTE'], observed=True)
              .agg(esp=('esperado', 'sum'), esp_1a=('esp_1a', 'sum'), ferias_1a=('ferias_1a', 'sum'),
                   prolongada=('perda_prolongada', 'sum'), curta=('perda_curta', 'sum'),
                   dias=('DATA', 'size')).reset_index())


def taxas(agg):
    """Taxas (fração) por bloco a partir de somas por integrante."""
    esp, esp1 = agg.esp.sum(), agg.esp_1a.sum()
    return {'ferias': agg.ferias_1a.sum() / esp1 if esp1 else np.nan,
            'prolongada': agg.prolongada.sum() / esp, 'curta': agg.curta.sum() / esp}


def bootstrap_taxas(agg, n=300, seed=7):
    """Reamostra integrantes com reposição e devolve as taxas de cada réplica."""
    rng = np.random.default_rng(seed)
    ids = agg.ID_INTEGRANTE.unique()
    g = agg.set_index('ID_INTEGRANTE')
    saida = []
    for _ in range(n):
        s = rng.choice(ids, size=len(ids), replace=True)
        saida.append(taxas(g.loc[s]))
    return pd.DataFrame(saida)


def reserva_curta(media, sigma2, n_posicoes, z=1.645):
    """Taxa de ausência curta que cobre o dia com nível de serviço dado por z.

    Variância da taxa diária = choque comum (sigma2, independe do tamanho) + binomial (p(1-p)/N).
    """
    return media + z * np.sqrt(sigma2 + media * (1 - media) / n_posicoes)


def quadro_extra(n_posicoes, ferias, prolongada, curta_nivel):
    """Quadro extra e perda total para n_posicoes a cobrir."""
    perda = ferias + prolongada + curta_nivel
    return n_posicoes * perda / (1 - perda), perda


def participacao_elegivel(tempo_meses, risco_faixa, meses=36):
    """Parcela do quadro com >= 12 meses de casa, mês a mês, mantendo o quadro estável.

    Parte da distribuição atual de tempo de casa e usa o risco de saída por faixa (Trilha 1):
    quem sai é reposto por uma contratação nova (tempo de casa zero). Liga a Trilha 2 à Trilha 1:
    alta rotatividade mantém muita gente abaixo de 1 ano, que ainda não tira férias.
    """
    faixa = np.searchsorted(FAIXAS_TEMPO_CASA[1:], np.arange(200), side='left')
    h = risco_faixa[faixa]
    n = np.bincount(np.minimum(np.asarray(tempo_meses, dtype=int), 199), minlength=200).astype(float)
    saida = []
    for _ in range(meses):
        saida.append(n[12:].sum() / n.sum())
        sai = n * h
        n = envelhecer(n - sai, sai.sum())
    return np.array(saida)

# ---------------------------------------------------------------- funções: escala
"""Funções da Trilha 3 (mudança da escala 6x1 para 5x2).

Para entregar o mesmo serviço, a folha varia na razão inversa da carga semanal de cada integrante:

    folha_nova / folha_atual = horas semanais atuais / horas semanais novas

Isso vale para a folha inteira de quem muda de escala (posições e cobertura de ausências), porque a
cobertura de ausências é proporcional à folha. O fator depende de o que acontece com a jornada:
  jornada diária mantida   -> cada pessoa passa a trabalhar 5 dias em vez de 6 (fator = dias atuais / 5)
  limite semanal de 40 h   -> fator = 44 / 40
  carga semanal mantida    -> o dia de trabalho fica mais longo (fator = 1)
"""


def perfil_jornada(ponto, min_dias=60):
    """Dias por semana, minutos por dia trabalhado e horas por semana de cada integrante.

    Usa só dias normais (situação A, sem feriado) para não confundir jornada com ausência.
    """
    n = ponto[(ponto.situacao == 'A') & (ponto.FERIADO == 0) & (ponto.grupo != 'Aprendiz')]
    trab = n.MINUTOS_PREVISTOS > 0
    g = (n.assign(trab=trab)
          .groupby(['ID_INTEGRANTE', 'empresa', 'grupo', 'escala'], observed=True)
          .agg(dias=('DATA', 'size'), dias_trab=('trab', 'sum'), minutos=('MINUTOS_PREVISTOS', 'sum'))
          .reset_index())
    g = g[(g.dias >= min_dias) & (g.dias_trab > 0)].copy()
    g['dias_sem'] = 7 * g.dias_trab / g.dias
    g['min_dia'] = g.minutos / g.dias_trab
    g['h_sem'] = 7 * g.minutos / g.dias / 60
    return g


def media_bootstrap(valores, n=500, seed=11, q=(.05, .95)):
    """Média e intervalo por reamostragem de integrantes."""
    v = np.asarray(valores, dtype=float)
    rng = np.random.default_rng(seed)
    m = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n)]
    return v.mean(), np.quantile(m, q[0]), np.quantile(m, q[1])


def fatores(dias_sem_6x1, horas_sem_6x1, dias_novos=5, horas_limite=40, horas_legais_atuais=44):
    """Fator de folha (folha nova / folha atual) em cada cenário de jornada."""
    return {'jornada diária mantida': dias_sem_6x1 / dias_novos,
            'limite semanal de 40 h': horas_legais_atuais / horas_limite,
            'carga semanal mantida': 1.0}


def quadro_adicional(folha, fator):
    """Integrantes a mais para o mesmo serviço, dada a folha atual de quem muda de escala."""
    return folha * (fator - 1)


def decompor_adicional(folha, fator, perda_ausencia):
    """Separa o adicional entre posições e cobertura de ausências (a folha = posições + cobertura).

    perda_ausencia: fração da folha que é cobertura de ausências (Trilha 2).
    """
    total = quadro_adicional(folha, fator)
    return {'posições': total * (1 - perda_ausencia), 'cobertura de ausências': total * perda_ausencia}


def efeito_rotatividade(folha, fator, turnover_pct, multiplicador, dias_vaga=30):
    """Efeito da mudança sobre saídas por ano e sobre vagas abertas.

    A folha maior gera mais saídas mesmo com a mesma taxa; o multiplicador representa uma mudança
    na taxa (a escala 5x2 pode reduzir ou aumentar a rotatividade; os dados não permitem estimá-lo).
    """
    saidas_antes = folha * turnover_pct / 100
    saidas_depois = folha * fator * turnover_pct / 100 * multiplicador
    return {'saídas/ano antes': saidas_antes, 'saídas/ano depois': saidas_depois,
            'variação de saídas/ano': saidas_depois - saidas_antes,
            'vagas abertas em média (depois)': saidas_depois * dias_vaga / 365,
            'vagas abertas em média (antes)': saidas_antes * dias_vaga / 365}

# ---------------------------------------------------------------- funções: integrado
"""Modelo integrado das três trilhas.

Base única: a posição, isto é, a pessoa escalada para trabalhar num dia (já descontada a folga semanal
da escala atual). Cada trilha entra como um multiplicador sobre ela:

    cobertura (T2)   a / (1 - a)                    a = fração da folha que é cobertura de ausências
    escala (T3)      s (f - 1) / (1 - a)            s = parcela da folha em 6x1, f = fator de folha
    vagas (T1)       T * dias_vaga / 365 * folha    T = turnover anual, dias_vaga = prazo para repor

Folha necessária por posição = (1 + s (f - 1)) / (1 - a) * (1 + T * dias_vaga / 365).

Para a operação atual, a folha F (quadro ativo) já contém a cobertura: posições = F (1 - a), e a
cobertura dentro da folha é F * a. A mudança de escala soma F * s * (f - 1), sem depender de a.
"""


def multiplicadores(a, s, f, turnover_pct, dias_vaga=30):
    """Componentes da folha, em pessoas por posição (posição = 1)."""
    base = 1.0
    cobertura = a / (1 - a)
    escala = s * (f - 1) / (1 - a)
    sem_vagas = base + cobertura + escala
    vagas = sem_vagas * turnover_pct / 100 * dias_vaga / 365
    return {'posições': base, 'cobertura (T2)': cobertura, 'escala (T3)': escala,
            'vagas (T1)': vagas, 'total': sem_vagas + vagas}


def operacao_atual(folha, a, s, f, turnover_pct, dias_vaga=30):
    """Componentes em pessoas para a operação atual (folha = quadro ativo, que já inclui cobertura)."""
    posicoes = folha * (1 - a)
    m = multiplicadores(a, s, f, turnover_pct, dias_vaga)
    return {k: v * posicoes for k, v in m.items()}


def custo_anual(pessoas, salario_mensal, encargos=1.0):
    """Custo anual em R$: pessoas x salário x 12 x multiplicador de encargos (1,0 = só salário)."""
    return pessoas * salario_mensal * 12 * encargos

# ---------------------------------------------------------------- funções: rotatividade_eb
"""Trilha 1: encolhimento bayesiano empírico entre empresas e faixas calibradas fora da amostra.

Modelo gama-Poisson para a taxa de cada empresa, r_e = y_e / E_e, com y = (admissões + rescisões) / 2 e
E = soma do headcount mensal. Estimativa encolhida: (y_e + m b) / (E_e + b), com m a taxa do grupo e
b = m / tau2, tau2 a variância entre empresas (momentos).
"""


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


def backtest_eb(S, H, inicio='2019-12-01', fim='2026-09-01', empresas=(3, 5, 6, 7, 8)):
    """Projeção (média 36m e encolhida) x turnover realizado nos H meses seguintes, por origem."""
    lin = []
    for o in pd.date_range(inicio, pd.Timestamp(fim) - pd.DateOffset(months=H), freq='MS'):
        eb, _, _ = taxas_eb(S, o)
        for e in empresas:
            h = S[e][S[e].mes <= o]
            f = S[e][(S[e].mes > o) & (S[e].mes <= o + pd.DateOffset(months=H))]
            if len(f) < H or h.hc.tail(12).mean() < 20 or e not in eb.index:
                continue
            lin.append({'empresa': e, 'origem': o, 'H': H, 'real': turnover_janela(f) * 12 / H,
                        'media36': taxa_anual(h, 36), 'eb36': eb[e]})
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

# ---------------------------------------------------------------- funções: reserva_custo
"""Trilha 2: reserva para ausência curta escolhida pelo menor custo esperado.

Num dia com taxa de ausência p, escalam-se Q titulares e R reservas, e cada um falta com probabilidade p
(a reserva também falta). Postos descobertos: U = max(0, Q - S), S ~ Binomial(Q + R, 1 - p).
Custo diário em pessoa-dia: R + k E[U], com k o custo de um posto descoberto.
"""
from scipy.stats import binom


def deficit_dia(Q, R, p):
    """E[U] e P(U > 0) num dia com taxa de ausência p."""
    n = Q + R
    teto = int(min(n, n * p + 12 * (np.sqrt(n * p * (1 - p)) + 1)))
    j = np.arange(R, teto + 1)
    return (binom.sf(j, n, p).sum() if len(j) else 0.0), binom.sf(R, n, p)   # E[(D - R)+] = soma P(D > j)


def avaliar(Q, R, ps):
    """Média sobre os dias observados (ps): postos descobertos por dia, P(algum descoberto) e atendimento."""
    r = np.array([deficit_dia(Q, R, p) for p in ps])
    eu = r[:, 0].mean()
    return {'E_descobertos_dia': eu, 'P_algum_descoberto': r[:, 1].mean(), 'atendimento': 1 - eu / Q}


def reserva_otima(Q, ps, k, R0=0):
    """R inteiro de menor custo R + k E[U]. O custo é convexo em R, então basta avançar até ele subir."""
    melhor, R = None, R0
    while True:
        a = avaliar(Q, R, ps)
        custo = R + k * a['E_descobertos_dia']
        if melhor is not None and custo > melhor['custo_dia']:
            return melhor
        melhor = {'R': R, 'custo_dia': custo, **a}
        R += 1


def quadro_extra_pct(Q, R, a_planejada):
    """Folha por posição - 1, em %: Q + R escalados por dia, mais a cobertura planejada (férias e
    afastamentos prolongados), que é a fração a_planejada da folha."""
    return ((Q + R) / (1 - a_planejada) / Q - 1) * 100

# ---------------------------------------------------------------- funções: efeito_escala_rotatividade
def efeito_escala_rotatividade(DADOS="dados"):
    """Trilha 3: associação entre a escala praticada (6x1 x 5x2) e a taxa de desligamento.

    Poisson para a taxa mensal de saída por vínculo (offset: log dos meses de exposição; erros robustos),
    com a escala medida no ponto e controles para empresa, grupo de função e tempo de casa < 1 ano.

    Uso: python src/efeito_escala_rotatividade.py dados
    """
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


# ====================================================================================================
# AUDITORIA DOS DADOS (números da Seção 2 do relatório)
# ====================================================================================================
# ---------------------------------------------------------------- auditoria dos dados (números citados na Seção 2 do relatório)
def auditoria(DADOS='dados'):
    """Verificações de qualidade que orientaram a modelagem (Seção 2 do relatório)."""
    rd = lambda n, **k: pd.read_csv(f'{DADOS}/{n}.csv', sep=';', encoding='utf-8-sig', **k)
    i = rd('dim_integrante', parse_dates=['DATAADMISSAO', 'DATADEMISSAO'])
    b = rd('fBatidas', parse_dates=['DATA'])
    a = rd('fAfastamentos', parse_dates=['DT_INICIO', 'DT_FIM'])
    h = rd('dim_horario', usecols=['ID_HORARIO', 'ESCALA'])
    ini, fim = pd.Timestamp('2025-01-01'), pd.Timestamp('2026-09-30')
    out = {'vínculos': len(i), 'pessoas (CODPESSOA)': i.CODPESSOA.nunique(), 'linhas de ponto': len(b), 'afastamentos': len(a),
           'vínculos com ponto': b.ID_INTEGRANTE.nunique()}
    # cobertura do ponto entre os vínculos do período, por empresa
    per = i[(i.DATAADMISSAO <= fim) & (i.DATADEMISSAO.isna() | (i.DATADEMISSAO >= ini))].copy()
    per['tem_ponto'] = per.ID_INTEGRANTE.isin(set(b.ID_INTEGRANTE))
    cob = (per.groupby('CODCOLIGADA').tem_ponto.mean() * 100).round(0)
    # dependência entre ter ponto e empresa x escala (qui-quadrado e V de Cramér)
    per = per.merge(h, on='ID_HORARIO', how='left')
    tab = pd.crosstab(per.CODCOLIGADA.astype(str) + '|' + per.ESCALA.astype(str), per.tem_ponto).values
    esp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    qui2 = ((tab - esp) ** 2 / esp).sum()
    out['V de Cramér (ponto x empresa e escala)'] = round(np.sqrt(qui2 / tab.sum() / (min(tab.shape) - 1)), 2)
    # linhas de ponto fora do contrato
    x = b.merge(i[['ID_INTEGRANTE', 'DATAADMISSAO', 'DATADEMISSAO']], on='ID_INTEGRANTE', how='inner')
    out['linhas antes da admissão'] = int((x.DATA < x.DATAADMISSAO).sum())
    out['linhas depois da demissão'] = int((x.DATADEMISSAO.notna() & (x.DATA > x.DATADEMISSAO)).sum())
    # afastamentos duplicados e minutos impossíveis
    out['afastamentos duplicados'] = int(a.duplicated().sum())
    out['linhas com minutos impossíveis'] = int(((b.MINUTOS_TRABALHADOS < 0) | (b.MINUTOS_TRABALHADOS > 1440)).sum())
    # escala do cadastro x horário usado no dia
    y = b.merge(i[['ID_INTEGRANTE', 'ID_HORARIO']].rename(columns={'ID_HORARIO': 'H_CAD'}), on='ID_INTEGRANTE', how='inner')
    out['% das linhas de ponto com horário diferente do cadastro'] = round((y.ID_HORARIO != y.H_CAD).mean() * 100, 1)
    # duas fontes de férias: ponto (situação F) x afastamentos (FERIAS), dia a dia, nos vínculos com ponto
    f = a[a.CATEGORIA_EVENTO == 'FERIAS'].dropna(subset=['DT_FIM'])
    dias = pd.concat([pd.DataFrame({'ID_INTEGRANTE': r.ID_INTEGRANTE, 'DATA': pd.date_range(r.DT_INICIO, r.DT_FIM)}) for r in f.itertuples()])
    dias = dias.drop_duplicates().assign(af=1)
    z = b[['ID_INTEGRANTE', 'DATA', 'SITUACAO_NA_DATA']].merge(dias, on=['ID_INTEGRANTE', 'DATA'], how='left')
    p1, p2 = (z.SITUACAO_NA_DATA == 'F'), z.af.fillna(0).astype(bool)
    po = (p1 == p2).mean(); pe = p1.mean() * p2.mean() + (1 - p1.mean()) * (1 - p2.mean())
    out['kappa de Cohen (férias: ponto x afastamentos)'] = round((po - pe) / (1 - pe), 2)
    out['dias de férias em só uma das fontes'] = int((p1 != p2).sum())
    # primeiro registro de ponto dos admitidos no período: cai no dia 21 (o ponto segue o ciclo da folha)
    adm = i[(i.DATAADMISSAO >= ini) & (i.DATAADMISSAO <= fim)][['ID_INTEGRANTE']]
    prim = b.groupby('ID_INTEGRANTE').DATA.min().rename('prim').reset_index().merge(adm, on='ID_INTEGRANTE')
    out['% dos admitidos no período com 1º registro no dia 21'] = round((prim.prim.dt.day == 21).mean() * 100, 1)
    # dias de férias no ponto em sequências de mais de 30 dias (não são férias pela CLT)
    fz = b[b.SITUACAO_NA_DATA == 'F'][['ID_INTEGRANTE', 'DATA']].sort_values(['ID_INTEGRANTE', 'DATA'])
    nova = (fz.ID_INTEGRANTE != fz.ID_INTEGRANTE.shift()) | (fz.DATA.diff().dt.days != 1)
    fz['seq'] = nova.cumsum()
    tam = fz.groupby('seq').agg(n=('DATA', 'size'), vinc=('ID_INTEGRANTE', 'first'))
    longas = tam[tam.n > 30]
    out['% dos dias de férias do ponto em sequências > 30 dias'] = round(longas.n.sum() / len(fz) * 100, 1)
    out['vínculos com sequência de férias > 30 dias'] = longas.vinc.nunique()
    out['maior sequência de férias (dias)'] = int(tam.n.max())
    print(pd.Series(out).to_string()); print('\ncobertura do ponto (% dos vínculos do período), por empresa:'); print(cob.to_string())
    return out, cob
print("\n" + "=" * 80 + "\nAUDITORIA DOS DADOS\n" + "=" * 80)
auditoria("dados")


# ====================================================================================================
# PARTE B. TRILHA 1 — ROTATIVIDADE
# ====================================================================================================
print("\n" + "=" * 80 + "\nPARTE B. TRILHA 1 — ROTATIVIDADE\n" + "=" * 80)

# # 02 · Trilha 1 — Projeção da rotatividade por empresa
#
# **Pergunta:** qual a melhor estratégia para projetar a rotatividade (turnover) de cada empresa ao longo de
# um contrato de 3 anos?
#
# `Turnover (%) = [(Admissões + Rescisões) / 2] / Headcount × 100`
#
# Caminho deste notebook:
#
# 1. montar a série mensal de admissões, rescisões e headcount por empresa a partir do cadastro;
# 2. diagnosticar o que a série mostra (nível, tendência, sazonalidade, tempo de casa, tipo de saída);
# 3. **comparar estratégias de projeção por backtest** (projetar 12, 24 e 36 meses e conferir com o que
#    aconteceu);
# 4. escolher a estratégia e gerar a projeção com faixas de incerteza.
#
# O código reutilizável está em `src/rotatividade.py`.
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40)

integrantes = carregar_integrantes('dados/dim_integrante.csv')
serie = serie_mensal(integrantes)          # 2016-01 a 2026-09
S = {e: d.sort_values('mes').reset_index(drop=True) for e, d in serie.groupby('empresa')}
print(integrantes.shape, serie.shape, end="\n\n")

# ## 1. Definições e escolhas
#
# - **Unidade:** empresa (`CODCOLIGADA`).
# - **Admissão / rescisão:** início e fim de cada vínculo no cadastro (`dim_integrante`). Toda saída
#   conta, voluntária ou não. Recontratação gera nova admissão (o cadastro registra um vínculo novo).
# - **Headcount do mês:** vínculos que existiram em algum momento do mês (conforme o enunciado, "com
#   vínculo durante o período").
# - **Janela de dados:** o ponto e os afastamentos cobrem 21 meses, mas **o cadastro cobre todo o
#   histórico**. A Trilha 1 só precisa do cadastro, então usamos **jan/2016 a set/2026**, bem mais que 21
#   meses. Isso permite testar projeções contra o que realmente aconteceu.
# - **Outubro de 2026 fica de fora:** só há demissões de 01 a 13/10 e nenhuma admissão registrada, e isso
#   distorceria o mês.
# - **Turnover mensal anualizado:** o turnover de uma janela de N meses é calculado pela fórmula acima com
#   o headcount médio da janela, e anualizado por 12/N. **Não é** a fórmula aplicada a um ano inteiro (cujo
#   headcount inclui todos os vínculos do ano e dá valores menores); as duas definições são comparadas na
#   seção 1.
# turnover mensal anualizado por ano-calendario (taxa mensal com o headcount de cada mes x 12; 2026 = 9 meses)
# obs.: e a definicao usada na projecao; a formula do enunciado aplicada ao ano inteiro esta na celula seguinte
linhas = []
for e, d in S.items():
    for ano, g in d.groupby(d.mes.dt.year):
        t = turnover_janela(g) * 12 / len(g)
        linhas.append({'empresa': e, 'ano': ano + 0, 'turnover_anual_%': t, 'hc_medio': g.hc.mean()})
anual = pd.DataFrame(linhas)
print(anual.pivot(index='ano', columns='empresa', values='turnover_anual_%').round(1), end="\n\n")

# **Duas definições, dois números.** A fórmula do enunciado depende do período: aplicada a **um mês** e
# multiplicada por 12 (*turnover mensal anualizado*, a definição usada na projeção), o headcount é o do mês;
# aplicada a **um ano inteiro**, o headcount passa a ser todos os vínculos que existiram no ano, inclusive
# quem entrou e saiu, e o valor fica menor. Nas empresas com muita entrada e saída (6 e 8) a diferença é
# grande. A tabela abaixo mostra as duas lado a lado, mais a fórmula aplicada a uma janela de 36 meses, que
# é o horizonte do contrato.
# a mesma formula do enunciado em periodos diferentes
defs = []
for e in (3, 4, 5, 6, 7, 8):
    defs.append({'empresa': e,
                 'mensal anualizado, 36m (projeção) %': taxa_anual(S[e][S[e].mes <= pd.Timestamp('2026-09-01')], 36),
                 'fórmula no ano 2024 %': turnover_periodo(integrantes, e, '2024-01-01', '2024-12-31'),
                 'fórmula no ano 2025 %': turnover_periodo(integrantes, e, '2025-01-01', '2025-12-31'),
                 'fórmula jan-set/2026 (sem anualizar) %': turnover_periodo(integrantes, e, '2026-01-01', CORTE),
                 'fórmula na janela de 36m %': turnover_periodo(integrantes, e, '2023-10-01', CORTE)})
defs = pd.DataFrame(defs).set_index('empresa')
import os; os.makedirs('resultados', exist_ok=True)
defs.round(2).to_csv('resultados/trilha1_turnover_definicoes.csv', encoding='utf-8-sig')
print(defs.round(1), end="\n\n")
print(anual.pivot(index='ano', columns='empresa', values='hc_medio').round(0), end="\n\n")
if TEM_MPL:   # gráfico salvo em figuras/t1_01.png
    # turnover movel de 12 meses por empresa
    def movel12(d):
        return d.set_index('mes').apply(lambda x: x, axis=1) if False else (
            ((d.adm + d.resc).rolling(12).sum() / 2) / d.hc.rolling(12).mean() * 100).set_axis(d.mes)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for e, cor in ((5, 'C0'), (8, 'C1')):
        movel12(S[e]).loc['2017':].plot(ax=ax[0], label=f'empresa {e}', color=cor)
    ax[0].set_title('Turnover móvel 12 meses (%): empresas estáveis'); ax[0].legend()
    for e, cor in ((6, 'C2'), (3, 'C3'), (7, 'C4')):
        movel12(S[e]).loc['2019':].plot(ax=ax[1], label=f'empresa {e}', color=cor)
    ax[1].set_title('Turnover móvel 12 meses (%): empresas com mudanças de regime'); ax[1].legend()
    for a in ax: a.set_xlabel('')
    plt.tight_layout(); plt.savefig('figuras/t1_01.png', dpi=130, bbox_inches='tight'); plt.close('all')

# **Leitura:**
#
# - **Empresa 5** (a maior, ~3.000 vínculos): turnover baixo e estável, entre 5% e 12% ao ano (cerca de
#   10% nos últimos anos).
# - **Empresa 8**: turnover subiu de ~12–20% (2016–2021) para ~27–33% (2022 em diante) e se manteve nesse
#   patamar.
# - **Empresa 6**: quebras fortes de nível. O quadro cresceu de ~100 vínculos (2016) para ~835
#   (2023–2024), caiu para ~515 em 2025 e voltou a ~800 em 2026. O turnover foi de 13–17% (2016–2022) para
#   39–75% desde 2023.
# - **Empresas 3 e 7**: pequenas e em crescimento, com séries ruidosas.
# - **Empresa 1** não tem mais vínculos ativos (todos desligados até 2026) e a **4** nunca teve mais que
#   7. Veremos como tratá-las.

# ## 2. Diagnóstico: o que explica a rotatividade?
# Sazonalidade: indice do turnover mensal por mes do ano (1,00 = media da empresa), 2019-2026
linhas = []
for e in (5, 6, 8):
    d = S[e][S[e].mes >= '2019-01-01'].copy()
    d['to'] = (d.adm + d.resc) / 2 / d.hc * 100
    idx = d.groupby(d.mes.dt.month)['to'].mean() / d['to'].mean()
    linhas.append(idx.rename(f'empresa {e}'))
saz = pd.concat(linhas, axis=1).round(2)
saz.index.name = 'mês'
print(saz, end="\n\n")

# A sazonalidade é **fraca e sem padrão comum**: os índices ficam perto de 1,00 e os meses mais altos ou
# baixos mudam de empresa para empresa (por exemplo, nov/dez são baixos só na empresa 6). Não há ganho claro
# em modelar sazonalidade nesta base, e o backtest abaixo confirma que métodos simples já ficam no limite do
# que os dados permitem.
# Risco mensal de saida por faixa de tempo de casa (36 meses ate set/2026)
painel = painel_tempo_casa(integrantes)
origem = pd.Timestamp('2026-09-01')
risco = pd.DataFrame({f'empresa {e}': risco_por_faixa(painel, e, origem) * 100 for e in (5, 6, 8, 3, 7)},
                     index=ROTULOS_FAIXAS).round(1)
risco.index.name = 'meses de casa'
print(risco, end="\n\n")
if TEM_MPL:   # gráfico salvo em figuras/t1_02.png
    ax = risco[['empresa 5', 'empresa 6', 'empresa 8']].plot(kind='bar', figsize=(9, 3.8))
    ax.set_ylabel('% do grupo que sai no mês'); ax.set_title('Risco mensal de saída por tempo de casa'); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig('figuras/t1_02.png', dpi=130, bbox_inches='tight'); plt.close('all')

# **O tempo de casa importa muito.** Na empresa 6, quem está no 3º mês de casa tem ~22% de chance de sair
# naquele mês, contra ~1,7% para quem tem mais de 5 anos. Na empresa 8 o padrão é parecido, mas mais suave
# (9% no 3º mês contra 0,7% com mais de 5 anos). Na empresa 5 o risco é baixo em todas as faixas (0,3% a
# 3%). O pico no 3º mês sugere fim de contrato de experiência de 90 dias. Isso significa que **empresas que
# contratam muito (como a 6 em 2026) terão rotatividade alta nos meses seguintes**.
# Tipo de saida nos ultimos 36 meses (rescisoes de out/2023 a set/2026)
d = integrantes[(integrantes.DATADEMISSAO >= '2023-10-01') & (integrantes.DATADEMISSAO <= CORTE)]
tipo = d.groupby(['CODCOLIGADA', 'CATEGORIA_DESLIGAMENTO']).size().unstack(fill_value=0)
tipo['% voluntária'] = (tipo['VOLUNTARIO'] / tipo.sum(axis=1) * 100).round(0)
print(tipo, end="\n\n")

# A parcela voluntária varia bastante: ~52% na empresa 6, ~50% na 8 e ~23% na 5. Para o preço do contrato, a
# saída voluntária é decisão do integrante e a involuntária é decisão da empresa (inclui fim de contrato), o
# que dá à empresa algum controle sobre a segunda parte.

# ## 3. Backtest: qual estratégia projeta melhor?
#
# **Procedimento (origem móvel).** Para cada mês de origem, usamos só os dados até ali, projetamos o
# turnover dos próximos 12, 24 ou 36 meses e comparamos com o turnover realmente observado nesse período. O
# erro é medido em pontos percentuais (p.p.) de turnover anual.
#
# **Estratégias testadas** (todas projetam uma taxa constante, e variam em quanta história usam):
#
# | Estratégia | Ideia |
# |---|---|
# | Média 6, 12, 24, 36 e 60 meses | taxa média da janela recente |
# | Média móvel ponderada | pesos que decaem com a idade (meia-vida de 6 e 12 meses) |
# | Mistura 12m + 36m | meio a meio entre curto e longo prazo |
# | **Coortes por tempo de casa** | envelhece o quadro atual com o risco de saída por tempo de casa (seção 2) |
#
# O backtest considera apenas empresas e origens com headcount médio ≥ 20 (abaixo disso a taxa é puro ruído).
def ewma(w, meia_vida):
    r = ((w.adm + w.resc) / 2 / w.hc.replace(0, np.nan) * 100).fillna(0).values
    p = 0.5 ** (np.arange(len(r))[::-1] / meia_vida)
    return (r * p).sum() / p.sum() * 12

METODOS = {
    'média 6m':   lambda h: taxa_anual(h, 6),
    'média 12m':  lambda h: taxa_anual(h, 12),
    'média 24m':  lambda h: taxa_anual(h, 24),
    'média 36m':  lambda h: taxa_anual(h, 36),
    'média 60m':  lambda h: taxa_anual(h, 60),
    'ponderada (meia-vida 6m)':  lambda h: ewma(h.tail(60), 6),
    'ponderada (meia-vida 12m)': lambda h: ewma(h.tail(60), 12),
    'mistura 12m+36m': lambda h: .5 * taxa_anual(h, 12) + .5 * taxa_anual(h, 36),
}

def backtest(metodos, empresas=(3, 5, 6, 7, 8), H=12, inicio='2019-12-01'):
    linhas = []
    ultima_origem = pd.Timestamp('2026-09-01') - pd.DateOffset(months=H)
    for e in empresas:
        s = S[e]
        for o in pd.date_range(inicio, ultima_origem, freq='MS'):
            h = s[s.mes <= o]; f = s[(s.mes > o) & (s.mes <= o + pd.DateOffset(months=H))]
            if len(f) < H or h.hc.tail(12).mean() < 20: continue
            real = turnover_janela(f) * 12 / H          # turnover anual realizado no periodo
            for nome, fn in metodos.items():
                linhas.append({'empresa': e, 'origem': o, 'H': H, 'metodo': nome, 'prev': fn(h), 'real': real})
    r = pd.DataFrame(linhas); r['erro'] = r.prev - r.real; r['abs'] = r.erro.abs()
    return r

bt12 = backtest(METODOS, H=12)
resumo = bt12.groupby('metodo').agg(MAE_pp=('abs', 'mean'), vies_pp=('erro', 'mean')).sort_values('MAE_pp').round(2)
print(resumo, end="\n\n")
print(bt12.pivot_table(index='metodo', columns='empresa', values='abs', aggfunc='mean').round(1).loc[resumo.index], end="\n\n")
# regime recente (origens a partir de 2024) e erro relativo nas tres empresas grandes
rec = bt12[bt12.origem >= '2024-01-01'].groupby('metodo').agg(MAE_pp_recente=('abs', 'mean'), vies_pp_recente=('erro', 'mean'))
g3 = bt12[bt12.empresa.isin([5, 6, 8])].assign(rel=lambda x: x['abs'] / x.real).groupby('metodo').rel.mean().rename('erro_relativo_5_6_8')
print(pd.concat([rec, g3], axis=1).sort_values('MAE_pp_recente').round(3), end="\n\n")

# As estratégias ficam num intervalo estreito (MAE de ~6,2 a ~6,9 p.p. no conjunto das 5 empresas), com a
# mistura 12m+36m e a média 36m praticamente empatadas na frente. Todas têm **viés negativo** (subestimam),
# porque a rotatividade subiu ao longo do tempo na maior parte das empresas. O erro se concentra na
# **empresa 6** (~13–16 p.p.) e na **3** (~8–13 p.p.), que mudam de regime. Nas empresas 5, 7 e 8 o erro é
# de 1,4 a 5,7 p.p.
# Cohorts (tempo de casa) no backtest, H=12, empresas 3,5,6,8
linhas = []
for e in (3, 5, 6, 8):
    s = S[e]
    for o in pd.date_range('2020-12-01', '2025-09-01', freq='MS'):
        h = s[s.mes <= o]; f = s[(s.mes > o) & (s.mes <= o + pd.DateOffset(months=12))]
        if len(f) < 12 or h.hc.tail(12).mean() < 20: continue
        real = turnover_janela(f)
        risco_o = risco_por_faixa(painel, e, o)
        adm12 = h.tail(12).adm.mean()
        for modo, adm_m in (('coorte (repõe saídas)', None), ('coorte (admissões = média 12m)', adm12)):
            sim = simular_coorte(integrantes, h, e, o, risco_o, adm_mensal=adm_m)
            linhas.append({'empresa': e, 'origem': o, 'metodo': modo, 'prev': turnover_janela(sim), 'real': real})
        linhas.append({'empresa': e, 'origem': o, 'metodo': 'média 36m', 'prev': taxa_anual(h, 36), 'real': real})
btc = pd.DataFrame(linhas); btc['abs'] = (btc.prev - btc.real).abs(); btc['erro'] = btc.prev - btc.real
print(pd.concat([btc.groupby('metodo').agg(MAE_pp=('abs', 'mean'), vies_pp=('erro', 'mean')).round(2),
           btc.pivot_table(index='metodo', columns='empresa', values='abs', aggfunc='mean').round(1)], axis=1), end="\n\n")

# **O modelo de coortes não supera a média de 36 meses** (MAE maior ou igual em todas as empresas). O motivo
# é que o erro vem sobretudo de **eventos coletivos** (demissões em massa, perda de contrato, expansão), que
# não dependem do tempo de casa. Ele continua útil como **explicação** (por que a empresa 6 tem turnover
# alto) e como **referência de cenário**, mas não como estimador principal.
# Horizontes longos: 12, 24 e 36 meses a frente (media 12m vs 36m)
lista = [backtest({'média 12m': METODOS['média 12m'], 'média 36m': METODOS['média 36m']}, H=H) for H in (12, 24, 36)]
bt = pd.concat(lista)
print(bt.pivot_table(index=['H', 'metodo'], columns='empresa', values='abs', aggfunc='mean').round(1), end="\n\n")

# Mesmo com horizonte de 36 meses, a taxa constante mantém erro baixo nas empresas **5** (~1,6 p.p.), **7**
# (~3 p.p.) e **8** (~5–7 p.p.). A empresa 3 tem erro de ~7 p.p. em 24 meses (não há histórico para 36).
# **Na empresa 6 o erro chega a 26–28 p.p. em 36 meses.**: o futuro dessa empresa depende de decisões
# comerciais (ganho ou perda de contratos), e não do histórico de rotatividade.

# ## 4. Estratégia escolhida
#
# **Taxa média dos últimos 36 meses, por empresa, com faixa de incerteza calibrada pelo backtest.**
#
# Por quê:
#
# - é a mais simples entre as de melhor desempenho (empate técnico com a ponderada e com a mistura);
# - a melhor no regime recente (origens a partir de 2024) e a de menor erro relativo para as empresas grandes;
# - 36 meses = o horizonte do contrato, e é longo o bastante para não reagir a um mês atípico (como a
#   saída em massa de 106 integrantes na empresa 8 em jan/2026);
# - não depende de pressupostos de sazonalidade ou de tendência, que os dados não sustentam.
#
# **Regras para casos especiais:**
#
# | Empresa | Situação | Tratamento |
# |---|---|---|
# | 5, 6, 7, 8 | histórico longo e headcount ≥ 20 | média 36m da própria empresa |
# | 3 | 114 vínculos mas histórico curto de crescimento (backtest só até H=24) | média 36m da própria empresa, faixa calibrada com H=24 |
# | 4 | 7 vínculos, nunca teve mais que isso | **taxa agrupada das empresas pequenas (3, 4, 7)**; confiança baixa |
# | 1 | sem vínculos ativos (encerrada em 2026) | **sem projeção** |
# | 2 | não existe no cadastro | sem projeção |
#
# **Incerteza.** A faixa (P10 a P90, limitada por um teto) vem dos erros do próprio backtest: razão entre o
# turnover realizado e o projetado, em origens móveis. Para o **ano 1** usa-se o horizonte de 12 meses. Para
# o **contrato inteiro (3 anos)**, o horizonte de 36 meses (24 meses na empresa 3, que não tem histórico
# para 36).
def razoes(H, empresas):
    r = backtest({'m36': METODOS['média 36m']}, empresas=empresas, H=H)
    r['razao'] = r.real / r.prev
    return r

def faixa(r):
    q = r.razao.quantile([.1, .5, .9])
    return q.iloc[0], q.iloc[1], q.iloc[2]

r12 = razoes(12, (3, 5, 6, 7, 8)); r24 = razoes(24, (3, 5, 6, 7, 8)); r36 = razoes(36, (5, 6, 7, 8))
FAIXA_1A = {e: faixa(r12[r12.empresa == e]) for e in (3, 5, 6, 7, 8)}
FAIXA_3A = {e: faixa(r36[r36.empresa == e]) for e in (5, 6, 7, 8)}
FAIXA_3A[3] = faixa(r24[r24.empresa == 3])
pequenas12 = r12[r12.empresa.isin([3, 7])]; pequenas24 = r24[r24.empresa.isin([3, 7])]
FAIXA_1A[4] = faixa(pequenas12); FAIXA_3A[4] = faixa(pequenas24)

# teto: maior turnover movel de 12 meses ja observado na empresa (com headcount medio >= 20). Evita faixas fisicamente irreais
def teto(e):
    d = S[e][S[e].mes <= pd.Timestamp('2026-09-01')]
    t = ((d.adm + d.resc).rolling(12).sum() / 2) / d.hc.rolling(12).mean() * 100
    return t[d.hc.rolling(12).mean() >= 20].max()
TETO = {e: teto(e) for e in (3, 5, 6, 7, 8)}

# cenario estrutural (coortes, quadro de set/2026): se passar do teto historico, o teto sobe ate ele
ORIGEM = pd.Timestamp('2026-09-01')
COORTE_MAX = {}
for e in (3, 5, 6, 7, 8):
    h = S[e][S[e].mes <= ORIGEM]; risco_o = risco_por_faixa(painel, e, ORIGEM)
    COORTE_MAX[e] = max(turnover_janela(simular_coorte(integrantes, h, e, ORIGEM, risco_o)),
                        turnover_janela(simular_coorte(integrantes, h, e, ORIGEM, risco_o, adm_mensal=h.tail(12).adm.mean())))
TETO[4] = max(TETO[3], TETO[7]); COORTE_MAX[4] = max(COORTE_MAX[3], COORTE_MAX[7])
TETO_EFETIVO = {e: max(TETO[e], COORTE_MAX[e]) for e in TETO}

faixas = pd.DataFrame({'ano 1 (P10, P50, P90)': {e: tuple(round(x, 2) for x in v) for e, v in FAIXA_1A.items()},
                       '3 anos (P10, P50, P90)': {e: tuple(round(x, 2) for x in v) for e, v in FAIXA_3A.items()}})
faixas.index.name = 'empresa (razão realizado / projetado)'
faixas['teto histórico (móvel 12m máx.)'] = pd.Series(TETO).round(1)
faixas['coortes (máx. cenários)'] = pd.Series(COORTE_MAX).round(1)
faixas['teto usado'] = pd.Series(TETO_EFETIVO).round(1)
print(faixas, end="\n\n")

# Leitura: para a empresa 5, o turnover realizado ficou entre 0,86× e 1,37× do projetado (ano 1). Para a 6,
# entre 0,87× e 3,3×, ou seja, a incerteza é muito maior e **assimétrica para cima**.
#
# Duas regras para transformar isso em faixa:
#
# - **A base é o piso.** Para as empresas 6, 7 e 8 no horizonte de 36 meses, o P10 da razão fica acima de
#   1 (em mais de 90% das origens o realizado superou o projetado, porque a rotatividade vinha subindo). Não
#   usamos esse viés para "subir" a base, pois ele reflete a tendência do período e não há garantia de que
#   continue. Mas também não afirmamos que a rotatividade pode ficar abaixo da base nessas empresas.
# - **Teto.** Na empresa 6, o P90 vem sobretudo de uma única mudança de regime (2022–2023), quando a
#   empresa saltou de ~16% para ~57% de turnover. Multiplicar a base por esse P90 daria mais saídas que o
#   próprio quadro. Por isso a faixa alta é limitada ao **maior entre o turnover móvel de 12 meses já
#   observado na empresa e o cenário de coortes** (coluna "teto usado").

# ## 5. Projeção para o contrato de 3 anos
def taxa_grupo(empresas, meses=36):
    w = pd.concat([S[e][S[e].mes <= ORIGEM].tail(meses) for e in empresas])
    return (w.adm + w.resc).sum() / 2 / w.hc.sum() * 1200

linhas = []
for e in (1, 3, 4, 5, 6, 7, 8):
    h = S[e][S[e].mes <= ORIGEM]
    hc = float(h.hc.iloc[-1]); ativos = int(((integrantes.CODCOLIGADA == e) & (integrantes.DATAADMISSAO <= CORTE)
                                             & (integrantes.DATADEMISSAO.isna() | (integrantes.DATADEMISSAO > CORTE))).sum())
    if ativos == 0:
        linhas.append({'empresa': e, 'ativos_set26': 0, 'metodo': 'sem projeção (sem vínculos ativos)'}); continue
    if e == 4:
        base, metodo, conf = taxa_grupo([3, 4, 7]), 'taxa agrupada das empresas pequenas (3, 4, 7)', 'baixa'
    else:
        base, metodo = taxa_anual(h, 36), 'média 36m da empresa'
        conf = {5: 'alta', 8: 'média-alta', 7: 'média', 3: 'média-baixa', 6: 'baixa'}[e]
    f1, f3 = FAIXA_1A[e], FAIXA_3A[e]
    linhas.append({
        'empresa': e, 'ativos_set26': ativos, 'metodo': metodo, 'confianca': conf,
        'turnover_base_%': base,
        # a base é o piso da faixa: onde o realizado superou o projetado em >90% das origens, o P10 fica acima de 1
        # a faixa alta nunca passa do teto (maior turnover já observado ou projetado por coortes)
        'ano1_baixo_%': base * min(f1[0], 1), 'ano1_alto_%': min(base * max(f1[2], 1), max(TETO_EFETIVO[e], base)),
        'contrato_baixo_%': base * min(f3[0], 1), 'contrato_alto_%': min(base * max(f3[2], 1), max(TETO_EFETIVO[e], base)),
        'saidas_ano_base': base / 100 * ativos,
        'saidas_ano_alto_3a': min(base * max(f3[2], 1), max(TETO_EFETIVO[e], base)) / 100 * ativos,
    })
proj = pd.DataFrame(linhas).set_index('empresa')
print(proj.round(1), end="\n\n")
# Referencia estrutural (coortes) para o ano 1: o que o modelo de tempo de casa projeta a partir do quadro de set/2026
linhas = []
for e in (3, 5, 6, 7, 8):
    h = S[e][S[e].mes <= ORIGEM]
    risco_o = risco_por_faixa(painel, e, ORIGEM)
    rep = turnover_janela(simular_coorte(integrantes, h, e, ORIGEM, risco_o))
    a12 = turnover_janela(simular_coorte(integrantes, h, e, ORIGEM, risco_o, adm_mensal=h.tail(12).adm.mean()))
    linhas.append({'empresa': e, 'média 36m': taxa_anual(h, 36), 'coorte (repõe saídas)': rep, 'coorte (adm = média 12m)': a12})
print(pd.DataFrame(linhas).set_index('empresa').round(1), end="\n\n")

# O modelo de coortes (só referência) aponta, para a empresa 6, uma rotatividade no ano 1 **acima** da média
# de 36 meses (70–71% contra 61%): o quadro de set/2026 tem muita gente com pouco tempo de casa
# (contratações de 2026), e esse grupo tem risco alto de saída. Na empresa 8 o modelo fica praticamente
# igual à média (28% contra 28,5%) e, na 5, abaixo. É um argumento para olhar a **parte alta da faixa** na
# empresa 6.
if TEM_MPL:   # gráfico salvo em figuras/t1_03.png
    ok = proj.dropna(subset=['turnover_base_%'])
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(ok)); ax.bar(x, ok['turnover_base_%'], color='C0', alpha=.8, label='base (média 36m)')
    ax.errorbar(x, ok['turnover_base_%'], yerr=[ok['turnover_base_%'] - ok['contrato_baixo_%'], ok['contrato_alto_%'] - ok['turnover_base_%']],
                fmt='none', ecolor='k', capsize=4, label='faixa P10-P90 (contrato 3 anos)')
    ax.set_xticks(x); ax.set_xticklabels([f'empresa {e}' for e in ok.index]); ax.set_ylabel('turnover anual (%)')
    ax.set_title('Projeção de turnover por empresa'); ax.legend(); plt.tight_layout(); plt.savefig('figuras/t1_03.png', dpi=130, bbox_inches='tight'); plt.close('all')
# salva o resultado
import os; os.makedirs('resultados', exist_ok=True)
proj.round(2).to_csv('resultados/trilha1_projecao.csv', encoding='utf-8-sig')
print(proj.round(1)[['ativos_set26', 'turnover_base_%', 'contrato_baixo_%', 'contrato_alto_%', 'confianca']], end="\n\n")

# ## 6. Premissas e limitações
#
# - **Taxa constante.** A projeção assume que o nível de rotatividade dos últimos 36 meses continua nos
#   próximos 3 anos, sem tendência nem sazonalidade. O backtest mostra que isso funciona bem para as
#   empresas 5 e 8, e mal para a 6.
# - **Viés de subestimação.** Na maior parte do histórico e das empresas, o realizado ficou acima do
#   projetado (a mediana da razão realizado/projetado é maior que 1 em todas as empresas com horizonte de 36
#   meses, 5, 6, 7 e 8, e na 3 com 24 meses). A faixa é assimétrica para cima, e **para precificação a faixa
#   alta é a referência prudente** nas empresas de confiança baixa.
# - **Empresa 6.** A rotatividade dela depende de ganhar ou perder contratos. A faixa é larga porque a
#   incerteza é real, e mesmo o teto histórico é um valor extremo, não um cenário provável.
# - **Saídas esperadas por ano** supõem headcount constante igual ao de set/2026 (admissões repõem
#   saídas). Isso não é uma projeção de quadro.
# - **Escala 6x1 → 5x2 (Trilha 3).** A projeção é do cenário atual. O efeito da mudança de escala na
#   rotatividade precisa ser tratado na Trilha 3 e não está aqui.
# - **Transferências entre empresas** (motivo de admissão 02) existem sobretudo nas empresas 3 e 7 e
#   contam como admissão e rescisão. Não foram separadas.
# - **Backtest com origens sobrepostas.** As origens consecutivas compartilham meses, então as faixas
#   P10/P90 são aproximadas, e para horizontes longos há poucas observações independentes.


# ====================================================================================================
# PARTE C. TRILHA 2 — FOLGUISTAS E FERISTAS
# ====================================================================================================
print("\n" + "=" * 80 + "\nPARTE C. TRILHA 2 — FOLGUISTAS E FERISTAS\n" + "=" * 80)

# # 03 · Trilha 2 — Folguistas e feristas
#
# **Pergunta:** quantos integrantes a mais, por empresa, são necessários para manter o serviço nos dias
# previstos quando os titulares faltam, tiram férias, apresentam atestado ou se afastam?
#
# **Ideia central.** A *perda de carga* é a fração do trabalho programado que deixa de ser entregue por
# causa de ausências. Para entregar N posições, a folha precisa de `N / (1 − perda)` pessoas, porque quem
# está ausente também está na folha. Logo:
#
# `quadro extra = N × perda / (1 − perda)`
#
# A perda é separada em três blocos, porque cada um se comporta de um jeito e pede um tratamento diferente:
#
# | Bloco | O que é | Tratamento |
# |---|---|---|
# | **Férias** | planejada (30 dias por ano, só para quem tem 1 ano de casa ou mais) | taxa entre elegíveis × parcela do quadro elegível, ligada à rotatividade da Trilha 1 |
# | **Afastamento prolongado** | INSS, acidente de trabalho, aposentadoria por incapacidade, licenças | nível médio (é um estado persistente, conhecido após o início) |
# | **Ausência curta imprevista** | faltas, atestados de até 15 dias, abonos | nível médio **mais uma reserva**, calibrada pela variabilidade diária, para um nível de serviço de 95% |
#
# Caminho: classificar o ponto, limpar problemas de dados, estimar as taxas, dimensionar a reserva,
# preencher as células sem dados e calcular o quadro extra por empresa.
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 100)
from statistics import NormalDist

# ## 1. Dados
ponto_bruto = carregar_ponto('dados')
ponto_bruto['grupo'] = ponto_bruto.FUNCAO.map(grupo_funcao)
print(len(ponto_bruto), 'linhas de ponto (vínculo-dia), sem os dias já demitidos')
print(ponto_bruto.groupby('bloco_dia', dropna=False).size().rename('dias').to_frame().assign(
    significado=lambda x: [{'ferias': 'situação F (férias)', 'prolongada': 'INSS, acidente, aposentadoria por incapacidade, licenças',
                            'curta': 'doença de até 15 dias (situação P sem INSS)'}.get(i, 'dia sem situação de ausência') for i in x.index]), end="\n\n")
# quem tem ponto, frente ao quadro ativo de set/2026
integ = carregar_integrantes('dados/dim_integrante.csv')
extra = pd.read_csv('dados/dim_integrante.csv', sep=';', encoding='utf-8-sig', usecols=['ID_INTEGRANTE', 'ID_FUNCAO'])
fun = pd.read_csv('dados/dim_funcao.csv', sep=';', encoding='utf-8-sig', usecols=['ID_FUNCAO', 'FUNCAO'])
integ = integ.merge(extra, on='ID_INTEGRANTE').merge(fun, on='ID_FUNCAO'); integ['grupo'] = integ.FUNCAO.map(grupo_funcao)
ativos = integ[(integ.DATAADMISSAO <= CORTE) & (integ.DATADEMISSAO.isna() | (integ.DATADEMISSAO > CORTE))].copy()
ativos['tempo_m'] = ((CORTE - ativos.DATAADMISSAO).dt.days / 30.44).astype(int)
com_ponto = set(ponto_bruto.ID_INTEGRANTE)
cob = ativos.assign(com_ponto=ativos.ID_INTEGRANTE.isin(com_ponto)).groupby(['CODCOLIGADA', 'grupo']).agg(
    ativos=('com_ponto', 'size'), com_ponto=('com_ponto', 'sum'))
cob['% com ponto'] = (cob.com_ponto / cob.ativos * 100).round(0)
print(cob, end="\n\n")

# **O que os dados cobrem.** `fBatidas` tem uma linha por vínculo por dia (99,6% dos dias de vínculo das
# pessoas que aparecem lá) e `fAfastamentos` só contém integrantes que também estão em `fBatidas`. Ou seja,
# **toda a informação de ausência vem da mesma subpopulação de ~1.900 pessoas**.
#
# A situação do dia no ponto identifica a causa: `F` = férias, `P` = doença ou INSS, `T` = acidente de
# trabalho, `I`, `L`, `M`, `E` e `Y` = aposentadoria por incapacidade, licenças e outros. Nos dias de férias
# e de afastamento longo a jornada prevista é zero, então esses dias **não aparecem como falta** e precisam
# ser contados pela situação, usando a carga que o integrante teria no dia.
#
# **Cobertura frente ao quadro ativo de set/2026 (tabela acima).** Dos 4.727 ativos operacionais (sem
# aprendizes), **3.570 (76%) não têm nenhuma linha de ponto**: toda a empresa 5 (2.824) e os 746 motoristas
# de ônibus da empresa 8. As empresas 6 e 3 e as funções "demais" da 8 estão bem cobertas. Esse é o
# principal limite desta trilha e orienta o desenho do resultado.

# ## 2. Limpeza do ponto
ponto, rel = limpar_ponto(ponto_bruto)
print(rel, end="\n\n")
if TEM_MPL:   # gráfico salvo em figuras/t2_01.png
    # efeito da limpeza: falta diaria (% dos programados sem trabalhar e sem abono) nas empresas 3 e 8
    def falta_diaria(d, emp):
        x = d[(d.empresa == emp) & (d.MINUTOS_PREVISTOS > 0) & d.bloco_dia.isna() & (d.FERIADO == 0)]
        return (x.MINUTOS_FALTA >= .9 * x.MINUTOS_PREVISTOS).groupby(x.DATA).mean()

    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
    for a, emp in zip(ax, (3, 8)):
        falta_diaria(ponto_bruto, emp).loc['2025-01-01':'2025-05-31'].plot(ax=a, color='C3', label='antes da limpeza')
        falta_diaria(ponto, emp).loc['2025-01-01':'2025-05-31'].plot(ax=a, color='C0', label='depois')
        a.set_title(f'Empresa {emp}: % de programados com falta total, por dia'); a.set_xlabel(''); a.legend()
    plt.tight_layout(); plt.savefig('figuras/t2_01.png', dpi=130, bbox_inches='tight'); plt.close('all')

# Dois problemas de dados foram encontrados, e ambos distorceriam as taxas:
#
# 1. **Início sem registro.** Nas empresas 3 e 8, até 20/01/2025 e 20/03/2025 respectivamente, quase 100%
#    dos dias programados aparecem como falta total (gráfico acima, em vermelho), e depois a taxa cai para
#    menos de 1%. O ponto dessas empresas evidentemente não registrava batidas nesse período. Sem a
#    correção, a ausência curta das funções "demais" da empresa 8 sairia em 9,5% em vez de 3,0%. Foram
#    removidas 14.131 linhas.
# 2. **Abono coletivo.** Em 6 dias (03 e 04/03/2025, 24 e 31/12/2025, 16 e 17/02/2026) mais de 15% da
#    carga de toda a população recebeu abono: Carnaval e vésperas de feriado. São previsíveis pelo
#    calendário, e não imprevisto. Foram removidas 4.588 linhas.
#
# Total removido: 18.719 de 593.008 linhas (3,2%).

# ## 3. Perda de carga por causa
agg = agregar_por_integrante(ponto)
op = agg[agg.grupo != 'Aprendiz']        # aprendizes ficam de fora (ver texto)
linhas = []
for (e, g), x in agg.groupby(['empresa', 'grupo']):
    t = taxas(x)
    linhas.append({'empresa': e, 'grupo': g, 'integrantes': x.ID_INTEGRANTE.nunique(), 'pessoa_dia': int(x.dias.sum()),
                   'férias (≥1 ano de casa)': t['ferias'] * 100, 'prolongada': t['prolongada'] * 100, 'curta': t['curta'] * 100})
tab = pd.DataFrame(linhas).set_index(['empresa', 'grupo']); print(tab.round(2), end="\n\n")

# **Leitura da tabela.**
#
# - **Motoristas de caminhão e veículos leves da empresa 6** (972 pessoas, a célula mais robusta): férias
#   8,7%, afastamento prolongado 5,8% (INSS pesa muito) e ausência curta 3,0%.
# - **Funções "demais" das empresas 6 e 8** (176 e 348 pessoas): prolongada de 2,5% a 3,1%, curta de 2,1%
#   a 3,0%.
# - **Empresa 3**: férias de apenas 4,1%, porque metade do quadro tem menos de 1 ano de casa (efeito
#   explicado abaixo).
# - **Empresas 1, 4 e 7**: amostras pequenas ou atípicas (a empresa 1 foi encerrada, a 4 tem 6 pessoas, a
#   7 tem 46 e curta de 15,3% por poucos eventos). Entram com peso baixo.
# - **Aprendizes** têm ausência curta de 27% a 35%: as horas de estudo do contrato de aprendizagem são
#   abonadas. Não é ausência a cobrir, então **aprendizes ficam fora do dimensionamento** (229 ativos).
# - **Motoristas de ônibus (empresa 6, 57 pessoas)**: prolongada de 11% por pouquíssimos casos longos.
#   Amostra fina.
# efeito do tempo de casa
x = ponto[ponto.grupo != 'Aprendiz'].copy()
x['faixa'] = pd.cut(x.anos_casa, [-1, .5, 1, 2, 5, 100], labels=['<6m', '6m-1a', '1-2a', '2-5a', '>5a'])
x['falta'] = np.where(x.bloco_dia.isna(), np.minimum(x.MINUTOS_PREVISTOS, x.MINUTOS_FALTA), 0)
tc = x.groupby('faixa', observed=True).apply(lambda g: pd.Series({
    'férias': g.perda_ferias.sum() / g.esperado.sum() * 100, 'prolongada': g.perda_prolongada.sum() / g.esperado.sum() * 100,
    'curta': g.perda_curta.sum() / g.esperado.sum() * 100, 'faltas (parte da curta)': g.falta.sum() / g.esperado.sum() * 100,
    'pessoa-dias': len(g)}), include_groups=False)
print(tc.round(2), end="\n\n")

# O **tempo de casa muda tudo**, e é o que liga esta trilha à Trilha 1:
#
# - **Férias:** praticamente zero abaixo de 1 ano de casa e de 8,0% a 8,4% depois, próximo dos 30 dias por
#   ano da lei (30/365 = 8,2%), mas veja a ressalva logo abaixo: parte desses dias não é férias. Por isso a
#   taxa de férias de uma empresa depende da parcela do quadro que já tem direito.
# - **Ausência curta:** 5,1% nos primeiros 6 meses contra 2,4% a 2,5% com mais de 2 anos. Só as faltas vão
#   de 3,1% para 0,5%.
# - **Afastamento prolongado:** cresce com o tempo de casa (0,6% abaixo de 6 meses, ~6,4% acima de 2
#   anos), porque os casos de INSS se concentram em quem está há mais tempo.
#
# Consequência: empresas com rotatividade alta têm **menos férias** (muita gente ainda sem direito) e **mais
# faltas** (muita gente nova).
# ressalva: sequencias de ferias no ponto mais longas que o limite legal (30 dias por periodo aquisitivo)
fb = pd.read_csv('dados/fBatidas.csv', sep=';', encoding='utf-8-sig',
                 usecols=['ID_INTEGRANTE', 'CODCOLIGADA', 'DATA', 'SITUACAO_NA_DATA'], parse_dates=['DATA'])
fe = fb[fb.SITUACAO_NA_DATA == 'F'].sort_values(['ID_INTEGRANTE', 'DATA'])
bloco = (fe.groupby('ID_INTEGRANTE').DATA.diff().dt.days != 1).cumsum()
dur = fe.groupby(bloco).DATA.transform('size')
seq = pd.DataFrame({'dias F no ponto': fe.groupby('CODCOLIGADA').size(),
                    'em sequências > 30 dias': fe[dur > 30].groupby('CODCOLIGADA').size(),
                    'em sequências > 60 dias': fe[dur > 60].groupby('CODCOLIGADA').size()}).fillna(0).astype(int)
seq.loc['total'] = seq.sum()
seq['% em > 30 dias'] = (100 * seq['em sequências > 30 dias'] / seq['dias F no ponto']).round(1)
print(f'{fe.ID_INTEGRANTE[dur > 30].nunique()} vínculos com "férias" contínuas de mais de 30 dias; maior sequência: {dur.max()} dias')
print(seq, end="\n\n")

# **Ressalva sobre a validação das férias.** Parte dos dias marcados como férias (`F`) no ponto está em
# sequências contínuas de mais de 30 dias, e uma parte em sequências de mais de 60, o que não é compatível
# com férias pela CLT (provavelmente são afastamentos longos registrados como férias). Por isso a
# coincidência da taxa entre elegíveis com 30/365 = 8,2% **não deve ser lida como validação**: a taxa de
# férias propriamente dita é provavelmente menor e a de afastamento prolongado, maior. **O quadro extra
# total quase não muda**, porque esses dias continuam sendo ausência que precisa de cobertura; muda só a
# divisão entre os blocos.
# validacao do mecanismo de ferias: empresa 6, motoristas. Prever ferias de jan-set/2026 com a taxa de 2025
# (so quem tem >=1 ano) x parcela elegivel de 2026, contra a taxa media de 2025 sem ajuste
m = ponto[(ponto.empresa == 6) & (ponto.grupo == 'Motorista (caminhão e leves)')].copy()
m = m[m.DATA.dt.month.between(3, 9) | (m.DATA.dt.year == 2026)]
def ferias_stats(g):
    el = g[g.elegivel_ferias]
    return pd.Series({'férias observada (% da carga)': g.perda_ferias.sum() / g.esperado.sum() * 100,
                      'taxa entre elegíveis': el.perda_ferias.sum() / el.esperado.sum() * 100,
                      'parcela elegível (% da carga)': el.esperado.sum() / g.esperado.sum() * 100})
per = {'mar-set/2025': m[(m.DATA >= '2025-03-01') & (m.DATA <= '2025-09-30')], 'jan-set/2026': m[m.DATA.dt.year == 2026]}
v = pd.DataFrame({k: ferias_stats(g) for k, g in per.items()}).T
v['previsto 2026 pelo mecanismo'] = np.nan
v.loc['jan-set/2026', 'previsto 2026 pelo mecanismo'] = v.loc['mar-set/2025', 'taxa entre elegíveis'] * v.loc['jan-set/2026', 'parcela elegível (% da carga)'] / 100
v.loc['jan-set/2026', 'previsto 2026 sem ajuste (taxa média de 2025)'] = v.loc['mar-set/2025', 'férias observada (% da carga)']
print(v.round(2), end="\n\n")

# **Validação do mecanismo de férias.** Na empresa 6 (motoristas), as férias caíram de 6,9% (mar-set/2025)
# para 4,6% (jan-set/2026). A razão não é mudança de comportamento: em 2026 só 53% da carga vem de quem tem
# 1 ano de casa ou mais, contra 85% em 2025 (muita contratação). Aplicando a taxa de 2025 entre elegíveis
# (8,1%) à parcela elegível de 2026, o mecanismo prevê **4,3%**, contra **4,6% observado**. Usar a média de
# 2025 sem esse ajuste preveria 6,9%, um erro de 2,3 p.p.

# ## 3b. Validação fora da amostra
# (1) validacao no tempo: estima com mar-dez/2025 e preve jan-set/2026, celula a celula
def por_integrante(d):
    return agregar_por_integrante(d[d.grupo != 'Aprendiz'])
treino = por_integrante(ponto[(ponto.DATA >= '2025-03-21') & (ponto.DATA <= '2025-12-31')])
teste = por_integrante(ponto[ponto.DATA >= '2026-01-01'])
CEL = [(6, 'Motorista (caminhão e leves)'), (6, 'Demais'), (8, 'Demais'), (3, 'Demais')]
linhas = []
for emp_, grp in CEL:
    a_tr = treino[(treino.empresa == emp_) & (treino.grupo == grp)]; a_te = teste[(teste.empresa == emp_) & (teste.grupo == grp)]
    t_tr, t_te = taxas(a_tr), taxas(a_te)
    el_tr, el_te = a_tr.esp_1a.sum() / a_tr.esp.sum(), a_te.esp_1a.sum() / a_te.esp.sum()
    ferias_obs_tr = t_tr['ferias'] * el_tr                      # férias observadas em 2025 (% da carga)
    ferias_obs_te = t_te['ferias'] * el_te
    comp = {'férias (mecanismo)': (t_tr['ferias'] * el_te, ferias_obs_te), 'férias (sem ajuste)': (ferias_obs_tr, ferias_obs_te),
            'prolongada': (t_tr['prolongada'], t_te['prolongada']), 'curta': (t_tr['curta'], t_te['curta'])}
    tot_mec = t_tr['ferias'] * el_te + t_tr['prolongada'] + t_tr['curta']; tot_sem = ferias_obs_tr + t_tr['prolongada'] + t_tr['curta']
    tot_obs = ferias_obs_te + t_te['prolongada'] + t_te['curta']
    n_tr = a_tr.ID_INTEGRANTE.nunique(); Z = n_tr / (n_tr + 100)             # metodo completo: credibilidade contra a referencia do treino
    ref_tr = taxas(treino)
    mix = lambda k: Z * (t_tr[k] if not np.isnan(t_tr[k]) else ref_tr[k]) + (1 - Z) * ref_tr[k]
    tot_cred = mix('ferias') * el_te + mix('prolongada') + mix('curta')
    linhas.append({'empresa': emp_, 'grupo': grp, 'integrantes teste': a_te.ID_INTEGRANTE.nunique(),
                   'perda observada 2026 (%)': tot_obs * 100, 'prevista c/ mecanismo (%)': tot_mec * 100, 'erro (p.p.)': (tot_mec - tot_obs) * 100,
                   'prevista sem ajuste (%)': tot_sem * 100, 'erro sem ajuste (p.p.)': (tot_sem - tot_obs) * 100,
                   'prevista c/ credibilidade (%)': tot_cred * 100, 'erro c/ credibilidade (p.p.)': (tot_cred - tot_obs) * 100,
                   **{f'{k}: previsto/obs. (p.p.)': f'{v[0]*100:.1f} / {v[1]*100:.1f}' for k, v in comp.items()}})
bt = pd.DataFrame(linhas).set_index(['empresa', 'grupo'])
print(bt[['perda observada 2026 (%)', 'prevista sem ajuste (%)', 'erro sem ajuste (p.p.)', 'prevista c/ mecanismo (%)', 'erro (p.p.)', 'prevista c/ credibilidade (%)', 'erro c/ credibilidade (p.p.)']].round(1), end="\n\n")

# **Validação no tempo.** Estimando com mar–dez/2025 e prevendo jan–set/2026, o método completo (taxa de
# cada célula combinada com a referência por credibilidade, e férias ajustadas pela parcela com direito em
# 2026) erra **de −0,3 a +2,2 p.p.** de perda total, com erro médio absoluto de **1,2 p.p.**, contra 2,6
# p.p. se não se ajusta o tempo de casa. Nas empresas 6 (caminhão: 13,5% previsto contra 12,7% observado) e
# 8 ("demais": 13,2% contra 13,5%) o ajuste de férias é o que melhora o resultado.
#
# A empresa 3 mostra por que a credibilidade existe: sem ela o erro é de −4,4 p.p., porque o treino de 2025
# não tem ninguém com 1 ano de casa, então a taxa própria de férias não existe e a de afastamento prolongado
# é zero. Com a credibilidade, a referência preenche o que a célula não tem, e o erro cai para +2,2. O custo
# disso aparece na empresa 6 "demais", em que a credibilidade piora um pouco (+1,6 contra +0,2 p.p.), porque
# a célula já tinha dado suficiente.
# (2) extrapolacao: preve uma celula com as taxas de todas as OUTRAS celulas (o que fazemos com a empresa 5)
op_agg = agregar_por_integrante(ponto[ponto.grupo != 'Aprendiz'])
linhas = []
for emp_, grp in CEL:
    alvo = op_agg[(op_agg.empresa == emp_) & (op_agg.grupo == grp)]
    outros = op_agg[~((op_agg.empresa == emp_) & (op_agg.grupo == grp))]
    pre, obs = taxas(outros), taxas(alvo)
    por_cel = {(e2, g2): taxas(x) for (e2, g2), x in outros.groupby(['empresa', 'grupo']) if x.ID_INTEGRANTE.nunique() >= 40}
    tot_cel = [sum(t.values()) for t in por_cel.values()]
    linhas.append({'célula omitida': f'{emp_} · {grp}', 'integrantes': alvo.ID_INTEGRANTE.nunique(),
                   'previsto (%)': sum(pre.values()) * 100, 'observado (%)': sum(obs.values()) * 100, 'erro (p.p.)': (sum(pre.values()) - sum(obs.values())) * 100,
                   'faixa das outras células (%)': f'{min(tot_cel)*100:.1f} a {max(tot_cel)*100:.1f}',
                   'dentro da faixa?': min(tot_cel) <= sum(obs.values()) <= max(tot_cel),
                   'férias / prolongada / curta (obs.)': ' / '.join(f'{obs[k]*100:.1f}' for k in ('ferias', 'prolongada', 'curta')),
                   'férias / prolongada / curta (prev.)': ' / '.join(f'{pre[k]*100:.1f}' for k in ('ferias', 'prolongada', 'curta'))})
loco = pd.DataFrame(linhas).set_index('célula omitida')
rmse_todas = float(np.sqrt((loco['erro (p.p.)'] ** 2).mean())); rmse_sem3 = float(np.sqrt((loco['erro (p.p.)'].iloc[:3] ** 2).mean()))
import os; os.makedirs('resultados', exist_ok=True)
pd.DataFrame({'rmse_pp': [rmse_todas, rmse_sem3], 'celulas': ['todas (4)', 'sem a empresa 3 (3)']}).to_csv('resultados/trilha2_erro_extrapolacao.csv', index=False, encoding='utf-8-sig')
print(f'erro de extrapolação (RMSE): {rmse_todas:.1f} p.p. com as 4 células, {rmse_sem3:.1f} p.p. sem a empresa 3')
print(loco.round(1), end="\n\n")

# **Extrapolação (é o que fazemos com a empresa 5).** Prever cada célula usando só as outras erra −2,9, +3,7
# e +1,9 p.p. de perda (caminhão da empresa 6, "demais" das empresas 6 e 8) e **+9,6 p.p. na empresa 3**,
# que tem taxas bem menores (prolongada de 0,3% e férias de 4,1% mesmo entre elegíveis). O erro quadrático
# médio é de **5,4 p.p.** com as 4 células e de 2,9 p.p. sem a empresa 3.
#
# Duas conclusões. **(1)** A faixa baixo–alto que usamos para células sem dado (cerca de ±1,6 p.p. de perda
# na empresa 5) é **estreita demais** frente ao erro de extrapolação observado. **(2)** O erro vem sobretudo
# de heterogeneidade real entre empresas, que só o dado próprio elimina. O modelo integrado (notebook 05)
# incorpora esse erro como uma fonte de incerteza separada.

# ## 4. Ausência curta: variabilidade diária e nível de serviço
o = ponto[ponto.grupo != 'Aprendiz']
dia = o.groupby('DATA').agg(c=('perda_curta', 'sum'), e=('esperado', 'sum'), n=('ID_INTEGRANTE', 'nunique'))
dia = dia[dia.n >= 100]; dia['p'] = dia.c / dia.e
dia = dia[dia.e > dia.e.median() * .5]                     # dias com jornada relevante (tira domingos e folgas coletivas)
p_med, n_med = dia.p.mean(), dia.n.mean()
var_obs, var_bin = dia.p.var(), p_med * (1 - p_med) / n_med
SIGMA2 = max(0, var_obs - var_bin)
dia[['p', 'n']].to_csv('resultados/trilha2_ausencia_curta_diaria.csv', encoding='utf-8-sig')   # insumo do notebook 06
print(pd.Series({'dias analisados': len(dia), 'ausência curta média diária': p_med, 'P95 empírico': dia.p.quantile(.95), 'P99 empírico': dia.p.quantile(.99),
           'variância observada': var_obs, 'variância se fossem independentes (binomial)': var_bin, 'choque comum (desvio)': SIGMA2 ** .5}).round(4).to_frame('valor'), end="\n\n")
if TEM_MPL:   # gráfico salvo em figuras/t2_02.png
    ax = dia.p.mul(100).plot(figsize=(11, 3.2), lw=.8); ax.axhline(p_med * 100, color='k', ls='--', lw=1)
    ax.axhline(dia.p.quantile(.95) * 100, color='C3', ls=':', lw=1); ax.set_title('Ausência curta diária (% da carga) — população operacional'); ax.set_xlabel(''); ax.set_ylabel('%')
    plt.tight_layout(); plt.savefig('figuras/t2_02.png', dpi=130, bbox_inches='tight'); plt.close('all')
# reserva extra (p.p. acima da media) para cobrir o dia com nivel de servico escolhido
from statistics import NormalDist
niveis = [.90, .95, .99]
print(pd.DataFrame({f'{int(nv*100)}%': {f'{N} posições': (reserva_curta(p_med, SIGMA2, N, z=NormalDist().inv_cdf(nv)) - p_med) * 100 for N in (50, 500, 3000)}
              for nv in niveis}).round(2), end="\n\n")

# Se as faltas de cada pessoa fossem independentes, a taxa diária de ausência curta variaria pouco
# (variância de ~0,00004). A variância observada é de ~0,00025, ou seja, ~7 vezes maior: há **choques
# comuns** que atingem muita gente no mesmo dia (desvio de ~1,5 p.p.). Por isso não basta cobrir a média de
# 3,3%: o dia ruim (P95) chega a 5,6% e o P99 a 9,5%.
#
# A reserva é a distância entre a média e o dia que queremos cobrir (o nível de serviço). O choque comum não
# diminui com o tamanho da operação, então a reserva também não: ~1,9 a 2,1 p.p. para 90%, ~2,5 a 2,75 p.p.
# para 95% e 3,5 a 3,9 p.p. para 99%, tanto em 500 quanto em 3.000 posições. Só em operações muito pequenas
# (50 posições) a parte binomial pesa (+4,8 p.p. a 95%).
#
# O nível de serviço de 95% é o caso base. A sensibilidade a 90% e 99% está na seção 6.
# sensibilidade: dias programados, sem nenhuma batida e sem falta nem abono registrados (tratados como sem perda)
ponto_op = ponto[ponto.grupo != 'Aprendiz']
x = ponto_op[(ponto_op.MINUTOS_PREVISTOS > 0) & ponto_op.bloco_dia.isna() & (ponto_op.FERIADO == 0) &
             (ponto_op.MINUTOS_TRABALHADOS == 0) & (ponto_op.MINUTOS_FALTA == 0) & (ponto_op.MINUTOS_ABONO == 0)]
print(pd.Series({'dias': len(x), 'em segunda ou domingo (%)': x.DATA.dt.dayofweek.isin([0, 6]).mean() * 100,
           'empresa 6 (%)': (x.empresa == 6).mean() * 100,
           'carga que seria perdida se fossem faltas (p.p.)': x.MINUTOS_PREVISTOS.sum() / ponto_op.esperado.sum() * 100}).round(2).to_frame('valor'), end="\n\n")

# **Um ponto de atenção.** Há 13.211 dias programados, sem nenhuma batida e sem falta nem abono registrados.
# São tratados aqui como **sem perda**. Estão concentrados em sábados e domingos (54%) e **98,7% são da
# empresa 6**. A hipótese é banco de horas ou folga compensada: na empresa 6, a hora extra total (6,4
# milhões de minutos) é da mesma ordem que a carga desses dias (5,6 milhões), então o trabalho extra feito
# antes cobre esses dias.
#
# Mas é uma hipótese que os dados não confirmam. Se fossem faltas, a carga perdida subiria 3,05 p.p. no
# conjunto operacional (4,65 p.p. na empresa 6), e o quadro extra da empresa 6 iria de ~18% para ~25%. Fica
# registrado como risco.

# ## 5. Preenchimento das células sem dados e referência de quadro
ORIGEM = pd.Timestamp('2026-09-01')
painel = painel_tempo_casa(integ)
op_all = op
REF = taxas(op_all)                                              # referência geral (população operacional com ponto)
REF_B = taxas(op_all[op_all.grupo == 'Demais'])                  # cenário baixo (menos prolongadas)
REF_A = taxas(op_all[op_all.grupo.str.startswith('Motorista')])  # cenário alto (motoristas)
print(pd.DataFrame({'baixo (demais funções)': REF_B, 'base (todos operacionais)': REF, 'alto (motoristas)': REF_A}).mul(100).round(2), end="\n\n")
K = 100   # peso de credibilidade: a celula com n integrantes pesa n/(n+K) contra a referencia
rows = []
for (e, g), x in ativos[ativos.grupo != 'Aprendiz'].groupby(['CODCOLIGADA', 'grupo']):
    N = len(x)
    cell = op_all[(op_all.empresa == e) & (op_all.grupo == g)]; n = cell.ID_INTEGRANTE.nunique()
    risco = risco_por_faixa(painel, e, ORIGEM)
    elegivel = participacao_elegivel(x.tempo_m.values, risco, 36)
    s = [elegivel[:12].mean(), elegivel[12:24].mean(), elegivel[24:].mean()]
    Z = n / (n + K)
    if n:
        pt = taxas(cell); boot = bootstrap_taxas(cell, n=200)
        lo, hi = boot.quantile(.1), boot.quantile(.9)
        pt = {k: (v if not np.isnan(v) else REF[k]) for k, v in pt.items()}
        lo = lo.fillna(pd.Series(REF_B)); hi = hi.fillna(pd.Series(REF_A))
    else:
        pt, lo, hi = REF, pd.Series(REF_B), pd.Series(REF_A)
    mix = lambda own, ref: {k: Z * own[k] + (1 - Z) * ref[k] for k in ('ferias', 'prolongada', 'curta')}
    rows.append({'empresa': e, 'grupo': g, 'ativos': N, 'integrantes_com_dados': n, 'peso_dados': Z,
                 'pct_1ano_atual': (x.tempo_m >= 12).mean(), 's_ano1': s[0], 's_ano2': s[1], 's_ano3': s[2],
                 'base': mix(pt, REF), 'baixo': mix(lo, REF_B), 'alto': mix(hi, REF_A)})
cel = pd.DataFrame(rows)
vis = cel[['empresa', 'grupo', 'ativos', 'integrantes_com_dados', 'peso_dados', 'pct_1ano_atual', 's_ano1', 's_ano3']].copy()
vis['férias (≥1a)'] = cel.base.map(lambda b: b['ferias'] * 100); vis['prolongada'] = cel.base.map(lambda b: b['prolongada'] * 100)
vis['curta'] = cel.base.map(lambda b: b['curta'] * 100)
print(vis.set_index(['empresa', 'grupo']).round(2), end="\n\n")

# **Como cada célula (empresa × grupo de função) é preenchida.**
#
# - A taxa de cada célula é uma média ponderada entre a **própria taxa** e a **referência geral** (todos
#   os operacionais com ponto), com peso de credibilidade `n / (n + 100)`, onde `n` é o número de
#   integrantes da célula com ponto. A célula de caminhão da empresa 6 (972 pessoas) pesa 91% nos próprios
#   dados, a de motoristas de ônibus da empresa 6 (57) pesa 36%, e uma célula **sem nenhum dado** (empresa
#   5, ônibus da empresa 8) fica 100% na referência geral.
# - **Cenários.** Para as células sem dados, o cenário baixo usa as taxas das funções "demais" (menos
#   afastamento prolongado) e o alto usa as dos motoristas (mais prolongado). Para as com dados, entra
#   também o intervalo de reamostragem por integrante (P10 a P90).
# - **Férias e tempo de casa.** A taxa de férias entre elegíveis é multiplicada pela parcela do quadro com
#   mais de 12 meses de casa, **projetada mês a mês** com o modelo de risco por tempo de casa da Trilha 1
#   (quadro estável, quem sai é reposto por um contratado novo). Isso depende da rotatividade: nos
#   motoristas de caminhão da empresa 6, a parcela elegível fica em ~48% no ano 1 e ~56% no ano 3, e na
#   empresa 5, em ~90% a ~96% nos dois anos.
# - **Quadro de referência.** O número de posições a cobrir é o quadro ativo operacional em 30/09/2026
#   (sem aprendizes), a mesma referência da Trilha 1.

# ## 6. Quadro extra por empresa
Z95 = NormalDist().inv_cdf(.95)
quadro_empresa = ativos[ativos.grupo != 'Aprendiz'].groupby('CODCOLIGADA').size()
def extra_celula(c, cenario, ano, z=Z95):
    t = getattr(c, cenario); s = [c.s_ano1, c.s_ano2, c.s_ano3][ano - 1]
    N_e = quadro_empresa[c.empresa]
    reserva = reserva_curta(t['curta'], SIGMA2, N_e, z) - t['curta']     # a reserva é do dia, comum à empresa
    ex, perda = quadro_extra(c.ativos, t['ferias'] * s, t['prolongada'], t['curta'] + reserva)
    return ex, perda

linhas = []
for c in cel.itertuples():
    for cen in ('baixo', 'base', 'alto'):
        for ano in (1, 2, 3):
            ex, perda = extra_celula(c, cen, ano)
            linhas.append({'empresa': c.empresa, 'grupo': c.grupo, 'ativos': c.ativos, 'cenario': cen, 'ano': ano, 'extra': ex, 'perda': perda})
res = pd.DataFrame(linhas)
por_empresa = res.groupby(['empresa', 'cenario', 'ano']).agg(extra=('extra', 'sum'), ativos=('ativos', 'sum')).reset_index()
por_empresa['extra_pct'] = por_empresa.extra / por_empresa.ativos * 100
vis = por_empresa.pivot(index='empresa', columns=['cenario', 'ano'], values='extra_pct').round(1)
vis = vis.reindex(columns=pd.MultiIndex.from_product([['baixo', 'base', 'alto'], [1, 2, 3]]))
print('Quadro extra em % das posições a cobrir (nível de serviço 95% para ausência curta)')
print(vis, end="\n\n")
hc = por_empresa.pivot(index='empresa', columns=['cenario', 'ano'], values='extra').round(0)
hc = hc.reindex(columns=pd.MultiIndex.from_product([['baixo', 'base', 'alto'], [1, 2, 3]]))
hc['posições'] = por_empresa.groupby('empresa').ativos.first()
print('Equivalente em integrantes, se o quadro ativo de set/2026 fossem só posições a cobrir')
print(hc, end="\n\n")

# **Resultado (cenário base, nível de serviço de 95%).** A tabela de percentuais mostra o quadro extra em %
# das posições a cobrir, por cenário e ano do contrato:
#
# | Empresa | Posições | Ano 1 | Ano 3 | Faixa baixo–alto (ano 3) | Confiança |
# |---|---|---|---|---|---|
# | 5 | 2.824 | 21,6% | 21,7% | 19,3% a 24,0% | **baixa** (nenhum dado próprio) |
# | 8 | 978 | 20,1% | 19,8% | 17,6% a 21,9% | média-baixa (76% do quadro sem dados) |
# | 6 | 760 | 18,0% | 18,7% | 16,2% a 21,4% | alta |
# | 3 | 113 | 14,8% | 16,6% | 14,6% a 18,6% | média |
# | 7 | 45 | 32,6% | 33,8% | 25,3% a 43,8% | baixa (amostra de 46 pessoas) |
# | 4 | 7 | 27,1% | 32,7% | 30,9% a 34,6% | muito baixa (7 pessoas) |
#
# Em número de integrantes (tabela seguinte), o cenário base soma **~970 integrantes extras no ano 1** e
# **~980 no ano 3**, com faixa de ~860 a ~1.080. Esses valores tomam o quadro ativo de set/2026 como se
# fossem posições a cobrir. Se esse quadro já inclui folguistas e feristas, o número de posições é menor e o
# extra em pessoas também. **O percentual é a medida mais robusta.** No notebook 05 o quadro ativo passa a
# ser tratado como a folha, que já contém a cobertura. Nessa leitura, que combina com a Trilha 3, o total em
# pessoas cai de ~970 (ativos × extra por posição) para ~810 (folha × perda por ausência). O percentual por
# posição não muda.
#
# **Por que o ano 1 e o ano 3 mudam pouco.** Só as férias variam ao longo do contrato, pela parcela do
# quadro com direito. Cresce nas empresas 3 e 6 (quadros jovens), é praticamente estável na 5 e cai
# levemente na 8.
# por empresa: contrato inteiro (media dos 3 anos) e decomposicao do cenario base
base = res[res.cenario == 'base'].copy()
dec = []
for c in cel.itertuples():
    t = c.base; N_e = quadro_empresa[c.empresa]
    res_curta = reserva_curta(t['curta'], SIGMA2, N_e, Z95) - t['curta']
    for ano in (1, 2, 3):
        s = [c.s_ano1, c.s_ano2, c.s_ano3][ano - 1]
        dec.append({'empresa': c.empresa, 'ano': ano, 'ativos': c.ativos, 'ferias': t['ferias'] * s, 'prolongada': t['prolongada'],
                    'curta_media': t['curta'], 'reserva': res_curta})
dec = pd.DataFrame(dec)
w = dec.assign(**{k: dec[k] * dec.ativos for k in ('ferias', 'prolongada', 'curta_media', 'reserva')}).groupby('empresa')[['ativos', 'ferias', 'prolongada', 'curta_media', 'reserva']].sum()
w['ativos'] = w.ativos / 3
for k in ('ferias', 'prolongada', 'curta_media', 'reserva'):
    w[k] = w[k] / 3 / w.ativos * 100
w['perda_total_%'] = w[['ferias', 'prolongada', 'curta_media', 'reserva']].sum(axis=1)
w['extra_%_base'] = w['perda_total_%'] / (100 - w['perda_total_%']) * 100
w.round(4).to_csv('resultados/trilha2_decomposicao_por_empresa.csv', encoding='utf-8-sig')   # insumo do notebook 06
print(w.round(1), end="\n\n")

# **Decomposição (média dos 3 anos, em % das posições).** Na empresa 5, a perda de 17,8% se divide em férias
# 7,6%, prolongada 4,4%, curta 3,3% e reserva 2,5%. Na empresa 8, 16,3% (6,4% + 4,1% + 3,2% + 2,6%) e, na 6,
# 15,5% (4,6% + 5,4% + 2,9% + 2,6%). A empresa 6 tem **menos férias** que as demais (4,6% contra 6,4% a
# 7,6%) por causa da rotatividade alta, mas **mais afastamento prolongado**.
#
# O afastamento prolongado (~4% a 5%) costuma ser coberto por contratação temporária e não por folguista
# fixo. Se o contrato não exigir cobrir esse bloco, o quadro extra cai em ~6 p.p. (de 21,6% para ~15,5% na
# empresa 5, por exemplo)
if TEM_MPL:   # gráfico salvo em figuras/t2_03.png
    cen3 = por_empresa.groupby(['empresa', 'cenario']).extra_pct.mean().unstack()[['baixo', 'base', 'alto']]
    fig, ax = plt.subplots(figsize=(9, 3.8)); x = np.arange(len(cen3))
    ax.bar(x, cen3.base, color='C0', alpha=.85, label='base')
    ax.errorbar(x, cen3.base, yerr=[cen3.base - cen3.baixo, cen3.alto - cen3.base], fmt='none', ecolor='k', capsize=4, label='faixa baixo–alto')
    ax.set_xticks(x); ax.set_xticklabels([f'empresa {e}' for e in cen3.index]); ax.set_ylabel('% de integrantes a mais'); ax.legend()
    ax.set_title('Quadro extra para cobrir ausências (média dos 3 anos)'); plt.tight_layout(); plt.savefig('figuras/t2_03.png', dpi=130, bbox_inches='tight'); plt.close('all')
# sensibilidade ao nivel de servico (base, empresa a empresa, media dos 3 anos)
sens = {}
for nv in (.90, .95, .99):
    r = []
    for c in cel.itertuples():
        for ano in (1, 2, 3):
            ex, _ = extra_celula(c, 'base', ano, z=NormalDist().inv_cdf(nv)); r.append({'empresa': c.empresa, 'ex': ex, 'N': c.ativos})
    r = pd.DataFrame(r).groupby('empresa').sum(); sens[f'{int(nv*100)}%'] = (r.ex / r.N * 100).round(1)
print(pd.DataFrame(sens), end="\n\n")

# A escolha do nível de serviço move o resultado em ~1 a 2 p.p. por degrau: na empresa 5, de 20,8% (90%)
# para 21,6% (95%) e 23,2% (99%). O efeito é bem menor que a incerteza sobre as taxas das células sem dados.
# salva
import os; os.makedirs('resultados', exist_ok=True)
saida = res.merge(cel[['empresa', 'grupo', 'peso_dados']], on=['empresa', 'grupo']); saida['extra_pct'] = saida.extra / saida.ativos * 100
saida.round(3).to_csv('resultados/trilha2_quadro_extra_por_celula.csv', index=False, encoding='utf-8-sig')
emp = por_empresa.copy(); emp.round(3).to_csv('resultados/trilha2_quadro_extra_por_empresa.csv', index=False, encoding='utf-8-sig')
print(emp[emp.cenario == 'base'].pivot(index='empresa', columns='ano', values='extra_pct').round(1), end="\n\n")

# ## 7. Limitações
#
# - **76% do quadro sem dado próprio.** A empresa 5 inteira e os ônibus da empresa 8 usam a referência
#   geral, que vem sobretudo de motoristas de caminhão da empresa 6 e de funções "demais". Ônibus e
#   cobradores podem ter taxas diferentes. A faixa baixo–alto cobre a dispersão entre grupos observados,
#   **mas o backtest de extrapolação (seção 3b) mostra que é estreita demais**: o erro de prever uma célula
#   sem dado próprio é de ~5 p.p. de perda (de 1,9 a 9,6). **É o maior risco do resultado e o dado que mais
#   o melhoraria**: o ponto e os afastamentos da empresa 5.
# - **Quadro de referência.** O quadro ativo de set/2026 é usado como proxy das posições. O percentual é a
#   medida robusta.
# - **Banco de horas.** 13.211 dias sem registro de falta, abono nem batida foram tratados como
#   compensados. Se fossem faltas, a empresa 6 subiria de ~18% para ~25%.
# - **Aprendizes** ficam fora (229 ativos). Se o contrato exigir cobertura deles, é preciso um tratamento
#   separado.
# - **Afastamento prolongado incluído.** Pelo enunciado, afastamentos entram. Mas o bloco pode ser coberto
#   por contratação temporária.
# - **Taxas constantes.** Fora o efeito do tempo de casa nas férias, as taxas são as médias de 2025–2026.
#   Não há tendência nem sazonalidade, e a base tem só 21 meses (~20 na empresa 3 e ~18 na 8, depois da
#   limpeza). A ausência curta é maior nos primeiros meses de casa (5,1%), efeito não modelado, que
#   subestima levemente as empresas de rotatividade alta (empresa 6).
# - **Choques raros** (greve, epidemia) não estão modelados: o episódio de ponto sem registro foi tratado
#   como dado ausente, e não como ausência real.
# - **Escala.** As taxas são do cenário atual. O efeito da mudança de 6x1 para 5x2 fica para a Trilha 3.


# ====================================================================================================
# PARTE D. TRILHA 3 — MUDANÇA DA ESCALA 6x1 PARA 5x2
# ====================================================================================================
print("\n" + "=" * 80 + "\nPARTE D. TRILHA 3 — MUDANÇA DA ESCALA 6x1 PARA 5x2\n" + "=" * 80)

# # 04 · Trilha 3 — Mudança da escala 6x1 para 5x2
#
# **Pergunta:** quantos integrantes a mais cada empresa precisa se a escala mudar de 6x1 para 5x2, mantendo
# o mesmo serviço?
#
# **Lógica.** Para entregar o mesmo trabalho, a folha varia na razão inversa da carga semanal de cada
# integrante:
#
# `folha nova / folha atual = horas semanais atuais / horas semanais novas`
#
# Isso vale para a **folha inteira** de quem muda de escala, não só para as posições: a cobertura de
# ausências (Trilha 2) é proporcional à folha, então já está contida no resultado. A Trilha 3 usa a Trilha 2
# para separar o adicional entre posições e cobertura, e a Trilha 1 para medir o efeito da folha maior sobre
# saídas e vagas.
#
# **Resumo do que o notebook mostra:**
#
# 1. **O que muda depende da jornada, e os dados mostram que isso não é trivial.** Hoje, quem está em 6x1
#    trabalha 5,98 dias × 7h08 = 42,7 h/semana, e quem está em 5x2 trabalha 5,04 dias × 8h33 = 43,0
#    h/semana. O 5x2 praticado é a mesma carga semanal em 5 dias. Se a mudança mantiver a carga semanal,
#    **não há adicional**. Se mantiver a jornada diária (a leitura do enunciado), a folha sobe **~20%**.
# 2. **Só 736 dos 4.727 ativos operacionais estão em 6x1 no cadastro** (634 na empresa 6). Outros 1.917
#    (empresa 5 em "Jornada 180 Mensais" e empresa 8 em "Revezamento") têm escala real desconhecida.
# 3. **Resultado:** adicional de **145 integrantes** (jornada diária mantida) para os 736 em 6x1, quase
#    todos na empresa 6 (125). Se as tripulações de jornada mensal e revezamento operarem em 6x1, o teto
#    sobe para **522**.
# 4. **Rotatividade:** o efeito é de segunda ordem frente ao da jornada. Os dados não permitem estimá-lo,
#    e o notebook trata como sensibilidade.
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40); print(pd.set_option('display.max_rows', 100), end="\n\n")

# ## 1. O que os dados dizem sobre as escalas
rd = lambda n, **k: pd.read_csv(f'dados/{n}.csv', sep=';', encoding='utf-8-sig', **k)
integ = rd('dim_integrante', usecols=['ID_INTEGRANTE', 'CODCOLIGADA', 'ID_FUNCAO', 'ID_HORARIO', 'JORNADAMENSAL', 'DATAADMISSAO', 'DATADEMISSAO'],
           parse_dates=['DATAADMISSAO', 'DATADEMISSAO'])
integ = (integ.merge(rd('dim_horario', usecols=['ID_HORARIO', 'ESCALA', 'HORARIO']), on='ID_HORARIO')
              .merge(rd('dim_funcao', usecols=['ID_FUNCAO', 'FUNCAO']), on='ID_FUNCAO'))
integ['grupo'] = integ.FUNCAO.map(grupo_funcao)
ativos = integ[(integ.DATAADMISSAO <= CORTE) & (integ.DATADEMISSAO.isna() | (integ.DATADEMISSAO > CORTE)) & (integ.grupo != 'Aprendiz')].copy()
print(f'{len(ativos)} ativos operacionais em 30/09/2026 (sem aprendizes)')
print(ativos.groupby(['CODCOLIGADA', 'ESCALA']).size().unstack(0).fillna(0).astype(int), end="\n\n")
# jornada mensal contratada (JORNADAMENSAL esta em minutos: 13200 = 220 h/mes = 44 h/semana)
jm = ativos.assign(horas_mes=pd.to_numeric(ativos.JORNADAMENSAL, errors='coerce') / 60)
print(jm.groupby(['ESCALA', 'horas_mes']).size().unstack(0).fillna(0).astype(int).loc[[220, 210, 180, 125, 100]], end="\n\n")

# `JORNADAMENSAL` está em minutos por mês. Todos os vínculos 6x1 têm 13.200 (220 h/mês, ou 44 h/semana). Os
# 5x2 são, na maioria, de 180 h/mês (1.288 vínculos), com minorias de 210 h e 220 h. As "Jornada 180
# Mensais" (empresa 5) têm 180 h e o revezamento da empresa 8 tem 220 h (557) ou 125 h (180).

# ### Dois sinais de que a escala do cadastro nem sempre é a real
# (a) empresa 5: tripulacoes de onibus 'em 5x2' com o mesmo horario generico
tr = ativos[(ativos.CODCOLIGADA == 5) & ativos.grupo.isin(['Motorista de ônibus', 'Cobrador'])]
a = tr.groupby(['ESCALA', 'HORARIO']).size().rename('ativos').reset_index().sort_values('ativos', ascending=False)
print('(a) empresa 5, motoristas de ônibus e cobradores:'); print(a.head(4), end="\n\n")
# (b) empresa 6, motorista de caminhao: demissoes com horario generico (demissoes em massa de 2023-2024)
cam = integ[(integ.CODCOLIGADA == 6) & (integ.FUNCAO == 'MOTORISTA/CAMINHAO')].assign(demitido=lambda x: x.DATADEMISSAO.notna())
b = cam.groupby('ESCALA').agg(vinculos=('demitido', 'size'), demitidos=('demitido', 'sum')); b['% demitidos'] = (b.demitidos / b.vinculos * 100).round(0)
x = cam[(cam.ESCALA == '5X2') & cam.demitido]
print('(b) empresa 6, motorista de caminhão, por escala do cadastro:'); print(b)
print('Horário dos', len(x), 'desligados em 5x2:'); print(x.HORARIO.value_counts().head(2), end="\n\n")

# - **(a) Empresa 5.** Os 1.194 motoristas de ônibus e cobradores "em 5x2" têm **todos o mesmo horário**,
#   "SEG A SEX 08:00–12:00 | 13:12–18:00". É um horário-padrão de cadastro, e não a escala real de uma
#   tripulação de ônibus. Os outros 1.091 estão em "Jornada 180 Mensais", que não informa os dias de folga.
#   **A escala real das tripulações da empresa 5 não está nos dados.**
# - **(b) Empresa 6.** Entre os motoristas de caminhão, 98% dos vínculos cadastrados em 5x2 estão
#   desligados (812 de 828), e 803 deles têm o mesmo horário "08:00 às 18:00 SEG-SEX". É o horário atribuído
#   nas demissões em massa de 2023–2024, não uma escala de trabalho.
#
# Duas consequências: a escala de ~1.900 pessoas (empresas 5 e 8) precisa ser tratada como incerta, e **não
# dá para estimar o efeito da escala sobre a rotatividade** comparando 5x2 e 6x1 no cadastro, porque a
# escala dos desligados é um artefato.

# ## 2. Quanto cada escala realmente trabalha
ponto, rel = limpar_ponto(carregar_ponto('dados'))
ponto['grupo'] = ponto.FUNCAO.map(grupo_funcao)
perfil = perfil_jornada(ponto)
res = perfil[perfil.escala.isin(['6X1', '5X2', '12X36'])].groupby('escala').agg(
    integrantes=('ID_INTEGRANTE', 'size'), dias_por_semana=('dias_sem', 'mean'), minutos_por_dia=('min_dia', 'mean'), horas_por_semana=('h_sem', 'mean'))
print(res.round(2), end="\n\n")
# 6x1 e 5x2 por empresa e grupo (celulas com pelo menos 20 integrantes)
c = perfil[perfil.escala.isin(['6X1', '5X2'])].groupby(['empresa', 'grupo', 'escala']).agg(
    integrantes=('ID_INTEGRANTE', 'size'), dias_sem=('dias_sem', 'mean'), min_dia=('min_dia', 'mean'), h_sem=('h_sem', 'mean'))
print(c[c.integrantes >= 20].round(2), end="\n\n")

# O ponto mostra a jornada praticada (só dias normais, sem férias, afastamento ou feriado):
#
# - **6x1:** 5,98 dias por semana × 428 minutos por dia = **42,7 h/semana**.
# - **5x2:** 5,04 dias por semana × 513 minutos por dia = **43,0 h/semana**.
# - Na mesma empresa e função o padrão se repete: na empresa 6, funções "demais" em 6x1 trabalham 43,2
#   h/semana e em 5x2, 42,7 h; na 8, 40,2 h em 6x1 e 43,3 h em 5x2.
#
# **O 5x2 que existe hoje é uma semana comprimida: mesma carga em menos dias, com dias mais longos (8h33
# contra 7h08).** Por isso a pergunta "quantas pessoas a mais" não tem resposta única: depende do que
# acontece com a jornada.

# ## 3. Cenários de jornada e fator de folha
d6 = perfil[perfil.escala == '6X1']
dias_m, dias_lo, dias_hi = media_bootstrap(d6.dias_sem)
horas_m, horas_lo, horas_hi = media_bootstrap(d6.h_sem)
fat = fatores(dias_m, horas_m)
fat_lo = fatores(dias_lo, horas_lo); fat_hi = fatores(dias_hi, horas_hi)
quadro = pd.DataFrame({'fator de folha': fat}).assign(**{'aumento de folha (%)': lambda x: (x['fator de folha'] - 1) * 100})
print(f'6x1 observado: {dias_m:.2f} dias/semana (IC90% {dias_lo:.2f} a {dias_hi:.2f}) e {horas_m:.1f} h/semana')
print(f'5x2 atual (mesmos dados): {res.loc["5X2", "dias_por_semana"]:.2f} dias/semana e {res.loc["5X2", "horas_por_semana"]:.1f} h/semana')
print(quadro.round(3), end="\n\n")

# | Cenário | O que acontece | Fator |
# |---|---|---|
# | **Jornada diária mantida** (caso base) | cada pessoa trabalha 5 dias de 7h08, em vez de 6, e a carga semanal cai de 42,7 h para ~35,7 h. É a leitura literal do enunciado ("mais folgas por integrante exige um quadro maior") | 5,98 / 5 = **1,197** (+19,7%) |
# | **Limite semanal de 40 h** | a carga semanal passa de 44 h legais para 40 h | 44 / 40 = **1,10** (+10%) |
# | **Carga semanal mantida** | o dia de trabalho passa para ~8h33, como já fazem os 5x2 atuais | **1,00** (sem adicional) |
#
# O intervalo de 90% do fator do caso base (por reamostragem de integrantes) é estreito (5,98 a 5,99 dias),
# porque a escala 6x1 é muito regular (5,98 dias trabalhados por semana, em média, entre 976 integrantes). A
# incerteza relevante não é estatística: é a **definição da jornada nova**, que o enunciado não fixa. O caso
# base segue o enunciado. O cenário de 40 h usa os limites legais (não observados nos dados).

# ## 4. Quadro adicional por empresa
# populacao A: 6x1 no cadastro (firme). populacao B: A + escalas variaveis (jornada mensal e revezamento), tratadas como 6x1
ativos['pop_A'] = ativos.ESCALA == '6X1'
ativos['pop_B'] = ativos.ESCALA.isin(['6X1', 'JORNADA MENSAL', 'REVEZAMENTO'])
folha = pd.DataFrame({'A: 6x1 no cadastro': ativos[ativos.pop_A].groupby('CODCOLIGADA').size(),
                      'B: A + jornada mensal + revezamento': ativos[ativos.pop_B].groupby('CODCOLIGADA').size(),
                      'ativos operacionais (total)': ativos.groupby('CODCOLIGADA').size()}).fillna(0).astype(int)
print(folha, end="\n\n")
cenarios = list(fat)
linhas = []
for emp, r in folha.iterrows():
    for pop, col in (('A', 'A: 6x1 no cadastro'), ('B', 'B: A + jornada mensal + revezamento')):
        for cen in cenarios:
            linhas.append({'empresa': emp, 'populacao': pop, 'cenario': cen, 'folha_6x1': r[col],
                           'adicional': quadro_adicional(r[col], fat[cen]),
                           'adicional_ic_baixo': quadro_adicional(r[col], fat_lo[cen]), 'adicional_ic_alto': quadro_adicional(r[col], fat_hi[cen])})
out = pd.DataFrame(linhas)
tab = out.pivot_table(index='empresa', columns=['populacao', 'cenario'], values='adicional').round(0)
tab = tab.reindex(columns=pd.MultiIndex.from_product([['A', 'B'], cenarios]))
tab.loc['total'] = tab.sum()
print(tab, end="\n\n")

# **Duas populações.** A população **A** são os vínculos com escala 6x1 no cadastro (736 ativos): é o
# conjunto firme. A população **B** soma a ela as 1.180 pessoas em "Jornada mensal" (empresa 5) e as 737 em
# "Revezamento" (empresa 8), **supondo que operem em 6x1**. Não há dado que confirme isso, então B é um
# teto.
#
# Integrantes a mais (tabela acima):
#
# | Empresa | Folha em 6x1 (A) | **A**, jornada diária mantida | A, limite de 40 h | **B**, jornada diária mantida | B, limite de 40 h |
# |---|---|---|---|---|---|
# | 6 | 634 | **125** | 63 | 125 | 63 |
# | 8 | 70 (A) / 807 (B) | **14** | 7 | 159 | 81 |
# | 7 | 30 | **6** | 3 | 6 | 3 |
# | 5 | 1 (A) / 1.181 (B) | 0 | 0 | 232 | 118 |
# | 3, 4 | 0 a 1 | 0 | 0 | 0 | 0 |
# | **Total** | 736 (A) / 2.653 (B) | **145** | 73 | **522** | 265 |
#
# Em **carga semanal mantida** o adicional é zero em qualquer população.
#
# O adicional em % dos ativos operacionais da empresa: empresa 6, **16,4%**; empresa 7, 13,1%; empresa 8,
# 1,4% (A) a 16,2% (B); empresa 5, ~0% (A) a 8,2% (B).
# decomposicao do adicional (cenario 'jornada diaria mantida', populacao A) em posicoes e cobertura de ausencias (Trilha 2)
t2 = pd.read_csv('resultados/trilha2_quadro_extra_por_empresa.csv')
t2 = t2[t2.cenario == 'base'].groupby('empresa').extra_pct.mean() / 100
perda = (t2 / (1 + t2)).rename('perda_ausencia')           # extra = perda / (1 - perda)  ->  perda = extra / (1 + extra)
dec = []
for emp, r in folha.iterrows():
    if r['A: 6x1 no cadastro'] == 0 or emp not in perda.index: continue
    d = decompor_adicional(r['A: 6x1 no cadastro'], fat['jornada diária mantida'], perda[emp])
    dec.append({'empresa': emp, 'folha 6x1': r['A: 6x1 no cadastro'], 'perda por ausência (T2)': perda[emp], **d,
                'adicional total': sum(d.values())})
print(pd.DataFrame(dec).set_index('empresa').round(2), end="\n\n")

# **Decomposição (Trilha 2).** Na empresa 6, a perda por ausência é de ~16%: dos 125 integrantes a mais
# (jornada diária mantida), **~105 são posições** e **~19 são cobertura de ausências**. A parte de cobertura
# aparece porque o fator multiplica a folha inteira, e a folha já contém os folguistas e feristas.
if TEM_MPL:   # gráfico salvo em figuras/t3_01.png
    fig, ax = plt.subplots(figsize=(9, 3.8))
    emps = tab.index[:-1]; x = np.arange(len(emps)); w = .26
    for k, cen in enumerate(cenarios):
        ax.bar(x + (k - 1) * w, tab.loc[emps, ('B', cen)].values, w, label=cen, alpha=.85)
        ax.bar(x + (k - 1) * w, tab.loc[emps, ('A', cen)].values, w, color='k', alpha=.55, label='população A (6x1 no cadastro)' if k == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels([f'empresa {e}' for e in emps]); ax.set_ylabel('integrantes a mais'); ax.legend(fontsize=8)
    ax.set_title('Quadro adicional por cenário (barra escura = população A; barra colorida = população B)'); plt.tight_layout(); plt.savefig('figuras/t3_01.png', dpi=130, bbox_inches='tight'); plt.close('all')

# ## 5. Efeito na rotatividade
# O cadastro nao permite estimar o efeito da escala na rotatividade (secao 1). Entra como sensibilidade sobre o turnover da Trilha 1.
t1 = pd.read_csv('resultados/trilha1_projecao.csv').set_index('empresa')['turnover_base_%']
linhas = []
for emp, r in folha.iterrows():
    if r['A: 6x1 no cadastro'] == 0 or emp not in t1.index or pd.isna(t1[emp]): continue
    for m in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
        e = efeito_rotatividade(r['A: 6x1 no cadastro'], fat['jornada diária mantida'], t1[emp], m)
        linhas.append({'empresa': emp, 'multiplicador da taxa': m, **e})
rot_tab = pd.DataFrame(linhas)
print(rot_tab.pivot(index='empresa', columns='multiplicador da taxa', values='variação de saídas/ano').round(0), end="\n\n")

# O enunciado diz que a mudança "também pode alterar a rotatividade". **Os dados não permitem medir esse
# efeito** (seção 1: a escala dos desligados é um artefato de cadastro), então ele entra como sensibilidade:
# um multiplicador sobre a taxa de rotatividade da Trilha 1, de 0,7 a 1,2.
#
# Com o multiplicador em 1,0 (mesma taxa), a folha maior já gera mais saídas por ano: na empresa 6 (turnover
# base de 61%), **+76 saídas por ano**, o que significa +76 contratações por ano para repor. Com a taxa 20%
# menor, o aumento cai para −16 (isto é, haveria menos saídas que hoje); com a taxa 20% maior, sobe para
# +169.
# vagas abertas em media (quantos postos ficam vagos, em equivalente de pessoas) com 30 dias para repor uma saida
print(rot_tab.pivot(index='empresa', columns='multiplicador da taxa', values='vagas abertas em média (depois)').round(1), end="\n\n")

# **Vagas abertas em média.** Supondo 30 dias para repor cada saída (premissa nossa, os dados não permitem
# estimar), a empresa 6 mantém hoje ~32 vagas abertas por dia. Com a folha 20% maior e a mesma taxa, passa a
# ~38.
# comparacao de ordem de grandeza (empresa 6): adicional de quadro x vagas abertas x variacao por mudar a taxa
e6 = rot_tab[rot_tab.empresa == 6].set_index('multiplicador da taxa')
print(pd.DataFrame({'adicional (jornada diária mantida)': quadro_adicional(folha.loc[6, 'A: 6x1 no cadastro'], fat['jornada diária mantida']),
              'vagas abertas se a taxa cair 20%': e6.loc[0.8, 'vagas abertas em média (depois)'],
              'vagas abertas se a taxa subir 20%': e6.loc[1.2, 'vagas abertas em média (depois)'],
              'vagas abertas hoje': e6.loc[1.0, 'vagas abertas em média (antes)']}, index=['integrantes']).T.round(1), end="\n\n")

# **Ordem de grandeza (empresa 6).** O adicional por horas é de ~125 integrantes. Mudar a taxa de
# rotatividade em 20% para mais ou para menos move as vagas abertas entre ~31 e ~46, ou seja, **algumas
# dezenas de pessoas contra 125**. Para compensar o aumento de vagas causado pela folha maior, a taxa
# precisaria cair ~17% (1/1,197). O efeito da rotatividade é real, mas de segunda ordem para o
# dimensionamento. Ele pesa mais no **custo** (admissões, treinamento), que o dimensionamento por headcount
# não captura.

# ## 6. As três trilhas juntas
# folha necessaria por empresa: ativos (hoje) + adicional da mudanca de escala (populacao A, jornada diaria mantida)
t1p = pd.read_csv('resultados/trilha1_projecao.csv').set_index('empresa')
conj = pd.DataFrame({'ativos operacionais': folha['ativos operacionais (total)'],
                     'folha em 6x1 (A)': folha['A: 6x1 no cadastro'],
                     'T3: adicional (A, jornada diária mantida)': [quadro_adicional(v, fat['jornada diária mantida']) for v in folha['A: 6x1 no cadastro']],
                     'T3: adicional (B, jornada diária mantida)': [quadro_adicional(v, fat['jornada diária mantida']) for v in folha['B: A + jornada mensal + revezamento']]})
conj['T3 A em % dos ativos'] = conj['T3: adicional (A, jornada diária mantida)'] / conj['ativos operacionais'] * 100
conj['T3 B em % dos ativos'] = conj['T3: adicional (B, jornada diária mantida)'] / conj['ativos operacionais'] * 100
conj['turnover T1 (%)'] = t1p['turnover_base_%'].reindex(conj.index)
print(conj.round(1), end="\n\n")

# A tabela reúne, por empresa, a folha atual, o adicional da Trilha 3 e o turnover da Trilha 1. A cobertura
# de ausências da Trilha 2 já está dentro do adicional (seção 4).
#
# Em síntese, o quadro adicional vem de **três fontes diferentes**: **(T2)** a cobertura de ausências sobre
# as posições, **(T3)** a folha a mais pela mudança de escala, que multiplica a folha inteira, e **(T1)** a
# reposição de saídas, que cresce com a folha. As empresas **6 e 7** são as mais afetadas pela mudança de
# escala (16% e 13% dos ativos), e a **8** e a **5** dependem de confirmar a escala real das tripulações.
import os; os.makedirs('resultados', exist_ok=True)
out.round(3).to_csv('resultados/trilha3_quadro_adicional.csv', index=False, encoding='utf-8-sig')
rot_tab.round(3).to_csv('resultados/trilha3_sensibilidade_rotatividade.csv', index=False, encoding='utf-8-sig')
print(conj.round(2).to_csv('resultados/trilha3_resumo_por_empresa.csv', encoding='utf-8-sig'), end="\n\n")

# ## 7. Limitações
#
# - **Escala real de 1.917 pessoas desconhecida.** As tripulações da empresa 5 ("Jornada 180 Mensais" e um
#   horário-padrão de 5x2) e da empresa 8 ("Revezamento") não informam dias de folga. O resultado firme (A)
#   as exclui, e o teto (B) as trata como 6x1. Se já trabalham 5 dias por semana, o adicional delas é zero.
#   **É a informação que mais melhoraria este resultado.**
# - **A jornada pós-mudança não é definida.** O resultado vai de 0 a +20% da folha 6x1, conforme a carga
#   semanal. O enunciado sugere o caso base, mas é uma escolha de contrato, não um dado.
# - **Rotatividade não estimável.** Sensibilidade, não previsão. O prazo para repor uma vaga (30 dias) é
#   premissa.
# - **Ausências inalteradas.** Supõe que as taxas de ausência da Trilha 2 não mudam com a escala. A
#   comparação por escala no ponto é contaminada por horários de passagem (afastados e desligados), então
#   não dá para testar.
# - **Demanda constante nos 7 dias.** O fator 6/5 supõe que o serviço é igual todos os dias da semana. Se
#   houver mais demanda em dias úteis, o efeito pode ser menor.
# - **Quadro fixo.** Usa a folha em 6x1 de set/2026 e não projeta crescimento (a empresa 6 cresceu 47%
#   desde jan/2025).
# - **Custo de transição.** Contratar e treinar ~125 pessoas leva tempo, e isso não está modelado.
#   Aprendizes ficam fora, como na Trilha 2.


# ====================================================================================================
# PARTE E. MODELO INTEGRADO E IMPACTO EM R$
# ====================================================================================================
print("\n" + "=" * 80 + "\nPARTE E. MODELO INTEGRADO E IMPACTO EM R$\n" + "=" * 80)

# # 05 · Modelo integrado e impacto financeiro
#
# Este notebook junta as três trilhas num único modelo, com **uma base comum**, e traduz o resultado em reais.
#
# **Base: a posição**, isto é, a pessoa escalada para trabalhar num dia, já descontada a folga semanal da
# escala atual. Cada trilha vira um multiplicador sobre ela:
#
# | Trilha | Componente | Fórmula (pessoas por posição) |
# |---|---|---|
# | **T2** | cobertura de ausências | `a / (1 − a)`, em que `a` é a parcela da folha que é cobertura |
# | **T3** | mudança de escala | `s (f − 1) / (1 − a)`, em que `s` é a parcela da folha em 6x1 e `f` o fator de folha |
# | **T1** | vagas abertas por rotatividade | `T × dias_vaga / 365 × folha`, em que `T` é o turnover anual |
#
# **Folha por posição = (1 + s (f − 1)) / (1 − a) × (1 + T × dias_vaga / 365)**
#
# **Por que isso resolve a inconsistência anterior.** A Trilha 2 media o extra *por posição* (`a / (1 −
# a)`), e a Trilha 3 multiplicava a *folha inteira*, que já inclui a cobertura. Eram bases diferentes. Aqui
# as duas partem da posição e o fator da T3 multiplica a folha que contém a cobertura. Para a **operação
# atual**, a folha F (quadro ativo, que inclui quem está de férias ou afastado) já contém a cobertura: as
# posições são `F × (1 − a)` e a cobertura dentro da folha é `F × a`. A mudança de escala soma `F × s × (f −
# 1)`, que não depende de `a`.
#
# **Duas premissas que não vêm dos dados** (e por isso aparecem como parâmetros): 30 dias para repor uma
# saída, e um multiplicador de 1,7 de encargos sobre o salário. A seção 5 mostra a sensibilidade.
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 100)
DIAS_VAGA = 30        # premissa: dias para repor uma saída (não estimável nos dados)
ENCARGOS = 1.7        # premissa da equipe: custo total do empregador / salário (não vem dos dados)

# ## 1. Insumos (lidos dos resultados das três trilhas)
t1 = pd.read_csv('resultados/trilha1_projecao.csv').set_index('empresa')
t2 = pd.read_csv('resultados/trilha2_quadro_extra_por_empresa.csv')
t3 = pd.read_csv('resultados/trilha3_quadro_adicional.csv')

# T2: perda por ausência a = e / (1 + e), em que e = quadro extra por posição. Média dos 3 anos do contrato.
e = t2.groupby(['empresa', 'cenario']).extra_pct.mean().unstack() / 100
a = e / (1 + e)
F = t2.groupby('empresa').ativos.first()                         # folha = quadro ativo operacional (set/2026)

# T3: fator de folha por cenario de jornada e parcela da folha em 6x1 (populacoes A e B)
ref = t3[(t3.populacao == 'A') & (t3.empresa == 6)].set_index('cenario')
f = (1 + ref.adicional / ref.folha_6x1).to_dict()
folha6 = t3.pivot_table(index='empresa', columns='populacao', values='folha_6x1', aggfunc='first')
s = folha6.div(F, axis=0).fillna(0)

# salario medio de referencia por empresa (ativos operacionais)
rd = lambda n, **k: pd.read_csv(f'dados/{n}.csv', sep=';', encoding='utf-8-sig', **k)
cad = (rd('dim_integrante', usecols=['ID_INTEGRANTE', 'CODCOLIGADA', 'ID_FUNCAO', 'SALARIO', 'DATAADMISSAO', 'DATADEMISSAO'], parse_dates=['DATAADMISSAO', 'DATADEMISSAO'])
       .merge(rd('dim_funcao', usecols=['ID_FUNCAO', 'FUNCAO']), on='ID_FUNCAO'))
cad = cad[(cad.DATAADMISSAO <= CORTE) & (cad.DATADEMISSAO.isna() | (cad.DATADEMISSAO > CORTE)) & ~cad.FUNCAO.str.upper().str.contains('APRENDIZ')]
sal = cad.groupby('CODCOLIGADA').SALARIO.mean()

emps = [e_ for e_ in F.index if e_ in t1.index and not pd.isna(t1.loc[e_, 'turnover_base_%'])]
ins = pd.DataFrame({'folha (ativos op.)': F, 'a base': a['base'], 'a baixo': a['baixo'], 'a alto': a['alto'],
                    's (A)': s['A'], 's (B)': s['B'], 'turnover base (%)': t1['turnover_base_%'],
                    'turnover alto (%)': t1['contrato_alto_%'], 'salário médio (R$)': sal}).loc[emps]
print('fator de folha por cenário de jornada:', {k: round(v, 3) for k, v in f.items()})
print(ins.round(3), end="\n\n")

# Todas as entradas vêm dos CSVs que as trilhas geraram, então o modelo integrado se atualiza se uma trilha
# mudar:
#
# - **a (T2):** o quadro extra por posição do cenário base (média dos 3 anos), convertido em `a = e / (1 +
#   e)`. Também lemos os cenários baixo e alto.
# - **F:** o quadro ativo operacional de set/2026 (sem aprendizes), que é a folha.
# - **s e f (T3):** a parcela da folha em 6x1 (população A: 6x1 no cadastro; B: inclui jornada mensal e
#   revezamento) e o fator de folha por cenário de jornada.
# - **T (T1):** o turnover base e o limite alto da faixa do contrato.
# - **Salário:** a média do **salário de referência** do cadastro, por empresa. É um valor de referência
#   por função, não o custo real efetivo.
#
# - **Erro de extrapolação da T2** (backtest do notebook 03): 5,4 p.p. de perda, aplicado só à parcela da
#   folha de cada empresa que **não tem dado próprio** (empresa 5: 100%; 8: 82%; 7: 68%; 6: 17%).

# ## 2. Folha por posição: um multiplicador, três componentes
def comp(emp, a_=None, s_pop='A', f_cen='jornada diária mantida', T=None, dias=DIAS_VAGA):
    return multiplicadores(a['base'][emp] if a_ is None else a_, s[s_pop][emp], f[f_cen],
                              t1.loc[emp, 'turnover_base_%'] if T is None else T, dias)

tabA = pd.DataFrame({emp: comp(emp) for emp in emps}).T * 100
tabB = pd.DataFrame({emp: comp(emp, s_pop='B') for emp in emps}).T * 100
print('Pessoas na folha para cada 100 posições (população A: 6x1 no cadastro; jornada diária mantida)')
print(tabA.round(1), end="\n\n")

# **População A, jornada diária mantida.** Para cada 100 posições, a folha tem de ter:
#
# | Empresa | Cobertura (T2) | Escala (T3) | Vagas (T1) | **Total** |
# |---|---|---|---|---|
# | 5 | 21,6 | 0,0 | 1,0 | **122,7** |
# | 8 | 19,9 | 1,7 | 2,8 | **124,4** |
# | 6 | 18,4 | 19,4 | 6,9 | **144,8** |
# | 7 | 33,4 | 17,5 | 1,7 | 152,6 |
# | 3 | 16,0 | 0,0 | 3,1 | 119,2 |
# | 4 | 30,5 | 3,7 | 3,0 | 137,2 |
#
# A cobertura é a parcela mais estável (16% a 33%). A escala só pesa onde há 6x1 no cadastro: **a empresa 6
# sai de 118 para 145 pessoas por 100 posições, sendo 18 de cobertura, 19 da mudança de escala e 7 de
# vagas.** As vagas são pessoas a mais para manter o serviço enquanto uma saída não foi reposta (premissa de
# 30 dias).
#
# As empresas 4 (7 pessoas) e 7 (45) têm amostra pequena; os números delas são indicativos.
print('Mesma conta com a população B (inclui jornada mensal e revezamento como 6x1)')
print(tabB.round(1), end="\n\n")

# **População B.** Se as tripulações de jornada mensal (empresa 5) e de revezamento (empresa 8) operarem em
# 6x1, a empresa 5 sobe de 123 para **133** pessoas por 100 posições e a empresa 8, de 124 para **142**. As
# outras não mudam. Essa diferença é a incerteza sobre a escala real dessas tripulações, e não uma incerteza
# estatística.

# ## 3. A operação atual em pessoas
atual = pd.DataFrame({emp: operacao_atual(F[emp], a['base'][emp], s['A'][emp], f['jornada diária mantida'], t1.loc[emp, 'turnover_base_%'], DIAS_VAGA)
                      for emp in emps}).T
atual['folha atual'] = F.loc[emps]
atual.loc['total'] = atual.sum()
print(atual.round(0), end="\n\n")

# Aplicando o modelo à operação atual, a folha de 4.727 pessoas se decompõe em **3.916 posições + 811 de
# cobertura de ausências**. Com a mudança de escala (população A, jornada diária mantida) e as vagas por
# rotatividade, a folha necessária passa a **4.967: +145 pela escala e +95 pelas vagas, ou seja, +240
# (+5,1%)**.
#
# A empresa 5 concentra a maior parte da folha (2.848 de 4.967) e quase não muda (+24, só vagas), porque
# nenhuma pessoa dela está em 6x1 no cadastro (população A). A empresa 6 é a que mais muda: de 760 para 929
# (+22%).
# reconciliacao com as notebooks anteriores
rec = pd.DataFrame({'T2 reportada (ativos × e)': F.loc[emps] * e['base'].loc[emps],
                    'T2 integrada (folha × a)': F.loc[emps] * a['base'].loc[emps],
                    'T3 (A, diária mantida)': [F[x] * s['A'][x] * (f['jornada diária mantida'] - 1) for x in emps]})
rec.loc['total'] = rec.sum()
print(rec.round(0), end="\n\n")

# **Reconciliação com os notebooks anteriores.** A Trilha 3 não muda: 145 integrantes. A Trilha 2 muda de
# **975 para 811** em número de pessoas. O que mudou foi só a **base** de contagem, e não o método:
#
# - No notebook 03, o extra por posição (`e`) foi multiplicado pelo quadro ativo, tratando-o como
#   *posições*: 975.
# - Aqui, o quadro ativo é tratado como a *folha*, que já contém a cobertura: `F × a` = 811.
#
# O percentual por posição da Trilha 2 (por exemplo, 21,6% na empresa 5) continua válido e é a medida que se
# aplica a uma licitação. A leitura adotada neste modelo é a que combina com a Trilha 3, e por isso passa a
# ser a de referência em pessoas.

# ## 4. De onde vem a incerteza
def total(emp, **k): return comp(emp, **k)['total'] * 100

# erro de extrapolacao da T2 (backtest, notebook 03): vale para a parcela da folha de cada empresa SEM dado proprio
cel2 = pd.read_csv('resultados/trilha2_quadro_extra_por_celula.csv')
cel2 = cel2[(cel2.cenario == 'base') & (cel2.ano == 1)]
sem_dado = (cel2.assign(p=cel2.ativos * (1 - cel2.peso_dados)).groupby('empresa').p.sum() / cel2.groupby('empresa').ativos.sum())
RMSE = pd.read_csv('resultados/trilha2_erro_extrapolacao.csv').rmse_pp.iloc[0] / 100
ext = (RMSE * sem_dado).reindex(emps)
print(f'erro de extrapolação da T2: {RMSE*100:.1f} p.p. de perda; parcela da folha sem dado próprio por empresa:', sem_dado.round(2).to_dict())
linhas = []
for emp in emps:
    base = total(emp)
    linhas.append({'empresa': emp, 'base (por 100 posições)': base,
        'a (T2): baixo': total(emp, a_=a['baixo'][emp]) - base, 'a (T2): alto': total(emp, a_=a['alto'][emp]) - base,
        'a (T2): extrapolação −1σ': total(emp, a_=a['base'][emp] - ext[emp]) - base, 'a (T2): extrapolação +1σ': total(emp, a_=a['base'][emp] + ext[emp]) - base,
        'turnover (T1): alto': total(emp, T=t1.loc[emp, 'contrato_alto_%']) - base,
        'população 6x1: B em vez de A': total(emp, s_pop='B') - base,
        'jornada: 40 h': total(emp, f_cen='limite semanal de 40 h') - base,
        'jornada: carga semanal mantida': total(emp, f_cen='carga semanal mantida') - base,
        'prazo de reposição: 15 dias': total(emp, dias=15) - base, 'prazo de reposição: 60 dias': total(emp, dias=60) - base})
torn = pd.DataFrame(linhas).set_index('empresa')
print(torn.round(1), end="\n\n")

# Cada coluna muda **uma** premissa por vez e mostra quantas pessoas por 100 posições o total se desloca em
# relação ao caso base:
#
# - **População 6x1 (A para B):** +10,1 na empresa 5 e +18,1 na empresa 8. É a escala real das
#   tripulações, que os dados não informam.
# - **Jornada pós-mudança:** na empresa 6, de −10,0 (limite de 40 h) a −20,4 (carga semanal mantida). É
#   uma definição de contrato.
# - **Cobertura de ausências (T2), erro de extrapolação:** de −7,6 a +8,7 na empresa 5, −6,3 a +7,0 na 8 e
#   −7,2 a +7,9 na 7. Na empresa 6, que tem dado próprio, é só ±1,6. Vem do backtest (notebook 03).
# - **Cobertura de ausências (T2), faixa de cenários:** ±2 a ±3 nas empresas grandes. É menor que o erro
#   de extrapolação, que é o número mais honesto para as empresas sem dado.
# - **Prazo de reposição (15 a 60 dias):** de −3,5 a +6,9 na empresa 6. É premissa.
# - **Rotatividade (T1, limite alto):** no máximo +2,1.
if TEM_MPL:   # gráfico salvo em figuras/integrado_01.png
    fatores = ['a (T2): faixa', 'a (T2): extrapolação', 'turnover (T1)', 'população 6x1', 'jornada', 'prazo de reposição']
    oscil = pd.DataFrame({
        'a (T2): faixa': torn['a (T2): alto'] - torn['a (T2): baixo'],
        'a (T2): extrapolação': torn['a (T2): extrapolação +1σ'] - torn['a (T2): extrapolação −1σ'],
        'turnover (T1)': torn['turnover (T1): alto'].abs(),
        'população 6x1': torn['população 6x1: B em vez de A'].abs(),
        'jornada': torn['jornada: carga semanal mantida'].abs(),
        'prazo de reposição': torn['prazo de reposição: 60 dias'] - torn['prazo de reposição: 15 dias']})
    ax = oscil[fatores].plot(kind='bar', stacked=False, figsize=(10, 3.8))
    ax.set_ylabel('oscilação (pessoas por 100 posições)'); ax.set_xlabel('empresa'); ax.set_title('Qual premissa mais mexe na folha, por empresa')
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig('figuras/integrado_01.png', dpi=130, bbox_inches='tight'); plt.close('all')

# O gráfico mostra que a fonte dominante de incerteza **depende da empresa**:
#
# - **Empresa 6** (dado próprio): domina a **jornada** pós-mudança (até −20 pessoas por 100 posições). A
#   estimação estatística pesa pouco.
# - **Empresas 5 e 8** (sem dado próprio em grande parte): três fontes de ordem parecida, a **escala real
#   das tripulações** (+10 e +18), o **erro de extrapolação das taxas de ausência** (±8 e ±7) e, na 8, a
#   população 6x1.
# - **Empresa 7** (45 pessoas): amostra pequena, com a faixa de ausência e o erro de extrapolação de ±8 a ±10.
#
# Em resumo, a maior parte da incerteza vem de informação que a SIAN tem e nós não: o **ponto da empresa
# 5**, a **escala real das tripulações** e a **jornada** que valerá após a mudança.

# ## 5. Impacto em R$
# custo anual da operação atual (R$ milhões), por componente; salário de referência médio da empresa
def milhoes(x): return x / 1e6
linhas = []
for emp in emps:
    pess = atual.loc[emp]
    for enc, rot_enc in ((1.0, 'só salário'), (ENCARGOS, f'com encargos (×{ENCARGOS})')):
        linhas.append({'empresa': emp, 'custo': rot_enc, **{k: milhoes(custo_anual(pess[k], sal[emp], enc)) for k in ['posições', 'cobertura (T2)', 'escala (T3)', 'vagas (T1)']}})
cus = pd.DataFrame(linhas)
cus['total'] = cus[['posições', 'cobertura (T2)', 'escala (T3)', 'vagas (T1)']].sum(axis=1)
sem = cus[cus.custo == 'só salário'].set_index('empresa').drop(columns='custo')
sem.loc['total'] = sem.sum()
print(sem.round(2), end="\n\n")

# Custo anual da operação atual por componente, usando o salário médio de referência de cada empresa. **Só
# salário** (sem encargos), em R$ milhões por ano: total de **R$ 151,1 milhões**, dos quais posições 120,1,
# cobertura de ausências 25,0, mudança de escala 3,4 e vagas 2,6. A empresa 5 responde por R$ 98,6 milhões
# (65%), e é a de dados menos confiáveis.
#
# Com o multiplicador de encargos de 1,7 (premissa da equipe), o total é de **R$ 256,8 milhões por ano**:
# posições 204,2, cobertura 42,6, escala 5,7 e vagas 4,4.
com = cus[cus.custo != 'só salário'].set_index('empresa').drop(columns='custo'); com.loc['total'] = com.sum()
print(f'Com encargos (×{ENCARGOS}, premissa), R$ milhões por ano'); print(com.round(2), end="\n\n")
# contrato de 3 anos (valores constantes), custo adicional frente a "so posicoes", e quanto cada trilha pesa
tot = com.loc['total']
resumo = pd.DataFrame({'R$ milhões/ano': tot, 'R$ milhões em 3 anos': tot * 3, '% do custo das posições': tot / tot['posições'] * 100}).round(1)
print(resumo, end="\n\n")

# **Atenção à leitura.** O "custo acima das posições" **não é** o aumento de custo em relação à operação de
# hoje: a cobertura de ausências (T2) já existe e já é paga na folha atual (são os ~811 integrantes que
# cobrem férias e afastamentos dentro dos 4.727). O que é **novo** em relação à folha atual é só a mudança
# de escala (T3) e as vagas por rotatividade (T1):
# acrescimo sobre a folha atual: a cobertura (T2) ja esta dentro da folha de hoje
novos = ['escala (T3)', 'vagas (T1)']
acr = pd.DataFrame({'pessoas': atual.loc[list(emps), novos].sum(),
                    'R$ mi/ano (só salário)': sem.loc['total', novos],
                    f'R$ mi/ano (encargos ×{ENCARGOS})': com.loc['total', novos]})
acr.loc['total'] = acr.sum()
acr[f'R$ mi em 3 anos (encargos ×{ENCARGOS})'] = acr[f'R$ mi/ano (encargos ×{ENCARGOS})'] * 3
acr[f'% da folha atual (encargos ×{ENCARGOS})'] = acr[f'R$ mi/ano (encargos ×{ENCARGOS})'] / (com.loc['total', ['posições', 'cobertura (T2)']].sum()) * 100
acr.round(2).to_csv('resultados/integrado_acrescimo_folha_atual.csv', encoding='utf-8-sig')
print(acr.round(1), end="\n\n")

# Em um contrato de 3 anos a valores constantes (sem reajuste, sem inflação e sem crescimento de quadro),
# com encargos de 1,7:
#
# - A **cobertura de ausências** custa **R$ 128 milhões** (20,8% do custo das posições). É o componente
#   que mais pesa.
# - A **mudança de escala** (população A, jornada diária mantida) custa **R$ 17 milhões** (2,8%).
# - As **vagas por rotatividade** custam **R$ 13 milhões** (2,1%).
# - Os três juntos somam cerca de **R$ 158 milhões** acima do custo das posições, ou 25,8%. **Isso não é o
#   aumento sobre a folha atual:** a cobertura já existe e já é paga hoje. O acréscimo real sobre a folha
#   atual é só escala + vagas: **+240 pessoas, cerca de R$ 10 milhões por ano ou R$ 30 milhões em 3 anos**
#   (4,1% da folha atual, com encargos de 1,7).
#
# A mudança de escala é pequena **na população A** (porque só 736 pessoas estão em 6x1 no cadastro). Na
# população B ela cresce bastante, e é o maior risco do resultado.
# reposicao de saidas (premissa de 1 salario por saida): custo anual de admissao e treinamento, fora do headcount
saidas = pd.Series({emp: atual.loc[emp, 'total'] * t1.loc[emp, 'turnover_base_%'] / 100 for emp in emps})
rep = pd.DataFrame({'saídas por ano': saidas, 'custo de reposição (R$ mi/ano, 1 salário por saída)': saidas * sal[emps] / 1e6})
rep.loc['total'] = rep.sum(); print(rep.round(2), end="\n\n")

# **Reposição de saídas.** A operação gera cerca de 1.194 saídas por ano. Com a premissa de um salário de
# custo por saída (recrutamento e treinamento, valor nosso), são **R$ 2,7 milhões por ano**, pequeno frente
# à folha. A rotatividade pesa mais pelas vagas em aberto (seção 3) que pelo custo de reposição direto, sob
# essa premissa.
# sensibilidade ao multiplicador de encargos (custo total das tres trilhas, R$ milhões por ano)
extra = sem.loc['total', ['cobertura (T2)', 'escala (T3)', 'vagas (T1)']].sum()
print(pd.DataFrame({f'encargos ×{x}': {'custo adicional (T1+T2+T3), R$ mi/ano': extra * x} for x in (1.0, 1.5, 1.7, 2.0)}).round(1), end="\n\n")
# custo da incerteza de extrapolacao (R$ milhoes por ano, com encargos): oscilacao do total por posicao x posicoes x salario
linhas = []
for emp in emps:
    posicoes = F[emp] * (1 - a['base'][emp])
    for lado, col in (('−1σ', 'a (T2): extrapolação −1σ'), ('+1σ', 'a (T2): extrapolação +1σ')):
        linhas.append({'empresa': emp, 'lado': lado, 'R$ mi/ano': custo_anual(posicoes * torn.loc[emp, col] / 100, sal[emp], ENCARGOS) / 1e6})
print(pd.DataFrame(linhas).pivot(index='empresa', columns='lado', values='R$ mi/ano').round(1), end="\n\n")

# O erro de extrapolação, que é o maior componente estatístico, custa aproximadamente **±R$ 11 milhões por
# ano** (com encargos) na empresa 5 e **±R$ 2 milhões** na empresa 8. Esse é o valor que o ponto da empresa
# 5 removeria. É grande frente ao custo de reposição direto (R$ 2,7 milhões por ano), e é a razão de pedir
# esse dado à organização.

# ## 6. Limitações
#
# - **A folha atual é tratada como adequadamente dimensionada.** O modelo supõe que o quadro ativo já
#   cobre as ausências (posições = F × (1 − a)). Se a operação atual estiver sub ou superdimensionada, as
#   posições reais diferem.
# - **Salário de referência, não custo real.** O cadastro traz um salário de referência por função.
#   Encargos (1,7), 13º, benefícios e horas extras estão no multiplicador, que é premissa, não dado. A seção
#   5 mostra a sensibilidade.
# - **A empresa 5 domina o custo** (65% do salário) e é a de menor confiança (sem ponto próprio).
# - **Valores constantes por 3 anos.** Sem reajuste, sem crescimento de quadro e sem a mudança de
#   composição que a Trilha 1 sugere (empresa 6 em expansão).
# - **Vagas como reserva de pessoas.** Tratar as vagas abertas como pessoas extras supõe que a empresa
#   mantém uma reserva. É premissa, com 30 dias para repor uma saída.
# - **Componentes combinados de forma simples.** A cobertura, a escala e a rotatividade se multiplicam sem
#   considerar interações entre elas (por exemplo, a mudança de escala alterar as ausências ou a
#   rotatividade).
# - **Escala e jornada.** O resultado da mudança de escala depende de uma decisão de contrato (jornada) e
#   de informação (escala real das tripulações) que não estão nos dados.
#
# - **Incerteza de extrapolação.** O erro de 5,4 p.p. de perda vem de 4 células (uma delas, a empresa 3,
#   muito diferente das outras). Com amostra tão pequena, é uma estimativa grosseira e pode estar
#   subestimada ou superestimada.
import os; os.makedirs('resultados', exist_ok=True)
pd.concat({'por_100_posicoes_A': tabA, 'por_100_posicoes_B': tabB}, names=['populacao', 'empresa']).round(3).to_csv('resultados/integrado_por_100_posicoes.csv', encoding='utf-8-sig')
atual.round(2).to_csv('resultados/integrado_operacao_atual_pessoas.csv', encoding='utf-8-sig')
cus.round(3).to_csv('resultados/integrado_custos.csv', index=False, encoding='utf-8-sig')
print(torn.round(3).to_csv('resultados/integrado_sensibilidade.csv', encoding='utf-8-sig'), end="\n\n")


# ====================================================================================================
# PARTE F. EXTENSÕES: FAIXAS VALIDADAS, BAYES EMPÍRICO, RESERVA POR CUSTO E ESCALA x ROTATIVIDADE
# ====================================================================================================
print("\n" + "=" * 80 + "\nPARTE F. EXTENSÕES: FAIXAS VALIDADAS, BAYES EMPÍRICO, RESERVA POR CUSTO E ESCALA x ROTATIVIDADE\n" + "=" * 80)

# # 06 · Extensões: faixas calibradas (T1), reserva por custo (T2), incerteza sem dados (T2) e escala ×
# rotatividade (T3)
#
# Quatro melhorias sobre os notebooks 02 a 05, cada uma validada e com resultado salvo em `resultados/`:
#
# 1. **T1:** encolhimento bayesiano empírico entre empresas e **faixas calibradas fora da amostra** (a
#    faixa do notebook 02 era calibrada com todas as origens, inclusive as futuras).
# 2. **T2:** a reserva para ausência curta passa a ser escolhida pelo **menor custo esperado**,
#    considerando que a própria reserva também falta, em vez de uma regra fixa de 95%.
# 3. **T2:** o erro de extrapolação medido (RMSE de 5,4 p.p.) é **propagado** para as células sem dado
#    próprio (empresa 5 e ônibus da empresa 8).
# 4. **T3:** estimativa da **associação entre a escala praticada e a taxa de desligamento** (modelo de
#    Poisson com exposição).
import warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
integrantes = carregar_integrantes('dados/dim_integrante.csv')
serie = serie_mensal(integrantes)
S = {e: d.sort_values('mes').reset_index(drop=True) for e, d in serie.groupby('empresa')}
proj = pd.read_csv('resultados/trilha1_projecao.csv', encoding='utf-8-sig').set_index('empresa')

# ## 1. T1 · Encolhimento entre empresas: o ponto central muda?
linhas = []
for H in (12, 24, 36):
    bt = backtest_eb(S, H)
    linhas.append({'H (meses)': H, 'origens': len(bt), 'MAE média 36m': (bt.media36 - bt.real).abs().mean(),
                   'MAE encolhida': (bt.eb36 - bt.real).abs().mean(),
                   'MAE emp. 3 (média)': (bt[bt.empresa == 3].media36 - bt[bt.empresa == 3].real).abs().mean(),
                   'MAE emp. 3 (encolhida)': (bt[bt.empresa == 3].eb36 - bt[bt.empresa == 3].real).abs().mean()})
print(pd.DataFrame(linhas).set_index('H (meses)').round(2), end="\n\n")

# O encolhimento empata com a média de 36 meses no conjunto e só ajuda a empresa 3 (a única pequena com
# histórico no backtest). **O ponto central do notebook 02 se mantém**; o encolhimento confirma que a
# escolha não é frágil.

# ## 2. T1 · As faixas cobrem o que dizem cobrir?
cob = []
for H in (12, 24):
    bt = backtest_eb(S, H)
    antiga = cobertura_faixa(bt)                       # P10-P90 por empresa, calibrada como no notebook 02
    nova = cobertura_lognormal(bt, z=1.2816)           # log-normal encolhida, só com o passado
    cob.append({'H (meses)': H, 'nominal': '80%', 'antiga, dentro da amostra': antiga.dentro.mean() * 100,
                'antiga, fora da amostra': antiga.fora.mean() * 100, 'nova, fora da amostra': nova.fora.mean() * 100})
print(pd.DataFrame(cob).set_index('H (meses)').round(0), end="\n\n")

# A faixa do notebook 02 parece cobrir 80% porque foi calibrada com as mesmas origens que avalia. **Sem
# olhar o futuro, cobre só 54% (12 meses) e 37% (24 meses).** A faixa nova (razão realizado/projetado em
# escala log, com centro e dispersão de cada empresa encolhidos para os do grupo) chega a 83% e ~70%. Com 36
# meses há poucas origens independentes e a calibração não é confiável (na empresa 6, a quebra de regime de
# 2020–2023 produz razões de até 5×), por isso usamos a faixa **anual**, limitada pelo maior turnover de 12
# meses já observado.
def teto(e):
    d = S[e]; t = ((d.adm + d.resc).rolling(12).sum() / 2) / d.hc.rolling(12).mean() * 100
    return t[d.hc.rolling(12).mean() >= 20].max()

bt12 = backtest_eb(S, 12); bt12['razao'] = bt12.real / bt12.media36
eb, _, _ = taxas_eb(S, pd.Timestamp('2026-09-01'))
fx = []
for e in (5, 8, 6, 3, 7, 4):
    lo, c, hi = faixa_lognormal(bt12[bt12.empresa == e].razao, bt12.razao, z=1.2816)
    base = proj.loc[e, 'turnover_base_%']; T = teto(e) if e != 4 else max(teto(3), teto(7))
    fx.append({'empresa': e, 'base (média 36m)': base, 'encolhida': eb.get(e, np.nan), 'faixa nova: P10': base * lo,
               'faixa nova: P90 (limitada ao teto)': min(base * hi, max(T, base)), 'teto histórico 12m': T,
               'faixa antiga': f"{proj.loc[e, 'contrato_baixo_%']:.1f} a {proj.loc[e, 'contrato_alto_%']:.1f}"})
fx = pd.DataFrame(fx).set_index('empresa')
fx.round(2).to_csv('resultados/trilha1_faixa_calibrada.csv', encoding='utf-8-sig')
print(fx.round(1), end="\n\n")

# O limite de cima quase não muda (já era o teto histórico). **O de baixo desce abaixo da base**: a regra "a
# base é o piso" do notebook 02 vinha da calibração dentro da amostra e não se sustenta fora dela. Para
# precificar, a leitura prudente continua sendo a parte alta da faixa, mas a faixa honesta é mais larga.

# ## 3. T2 · Reserva para ausência curta escolhida por custo
#
# Num dia com taxa de ausência curta p, escalam-se Q titulares + R reservas, e cada um falta com
# probabilidade p (a reserva também falta). Presentes: S ~ Binomial(Q + R, 1 − p); postos descobertos: U =
# max(0, Q − S). A distribuição de p é a **empírica** dos 551 dias observados (que já contém o choque
# comum), reescalada para a média de cada empresa. O custo diário, em pessoa-dia, é **R + k·E[U]**, em que k
# é o custo de um posto descoberto: ~1,5 se for coberto com hora extra (50%), ~3 se houver custo operacional
# adicional, ~10 se gerar multa contratual ou viagem não realizada.
dia = pd.read_csv('resultados/trilha2_ausencia_curta_diaria.csv', encoding='utf-8-sig')
dec = pd.read_csv('resultados/trilha2_decomposicao_por_empresa.csv', encoding='utf-8-sig').set_index('empresa')
p0 = dia.p.values
res = []
for e, r in dec.iterrows():
    perda = r['perda_total_%'] / 100; Q = int(round(r.ativos * (1 - perda)))
    mu = r.curta_media / 100; ps = p0 * mu / p0.mean()
    a_plan = (r.ferias + r.prolongada) / 100
    folha_regra = Q * (1 + r['extra_%_base'] / 100)                 # regra de 95% (notebook 03)
    R_regra = int(round(folha_regra * (1 - a_plan) - Q))
    av = avaliar(Q, R_regra, ps)
    res.append({'empresa': e, 'posições Q': Q, 'k': 'regra 95%', 'reserva R': R_regra, 'R/Q %': R_regra / Q * 100,
                'atendimento %': av['atendimento'] * 100, 'P(algum descoberto)': av['P_algum_descoberto'],
                'quadro extra %': r['extra_%_base']})
    for k in (1.5, 3, 10):
        o = reserva_otima(Q, ps, k, R0=max(0, int(Q * mu * .4)))
        res.append({'empresa': e, 'posições Q': Q, 'k': k, 'reserva R': o['R'], 'R/Q %': o['R'] / Q * 100,
                    'atendimento %': o['atendimento'] * 100, 'P(algum descoberto)': o['P_algum_descoberto'],
                    'quadro extra %': quadro_extra_pct(Q, o['R'], a_plan)})
res = pd.DataFrame(res)
res.round(3).to_csv('resultados/trilha2_reserva_por_custo.csv', index=False, encoding='utf-8-sig')
print(res.pivot(index='empresa', columns='k', values='quadro extra %').round(1), end="\n\n")
print(res.set_index(['empresa', 'k'])[['posições Q', 'reserva R', 'R/Q %', 'atendimento %', 'P(algum descoberto)']].round(3), end="\n\n")

# **Leitura.** A regra de 95% equivale a supor que um posto descoberto custa mais de 10 pessoas-dia. Se a
# operação cobre o descoberto com hora extra (k ≈ 1,5), a reserva ótima é bem menor e o quadro extra cai
# alguns pontos percentuais em todas as empresas, aceitando ~1% de postos-dia cobertos por hora extra. Se o
# contrato pune o descoberto (k ≈ 10), a reserva ótima fica próxima da regra. **O dado que decide é o custo
# de um posto descoberto no contrato**, e por isso o mostramos como parâmetro.

# ## 4. T2 · Incerteza das células sem dado próprio
#
# A validação do notebook 03 mediu o erro de extrapolar a perda para células sem dados: RMSE de 5,4 p.p.
# Propagamos esse erro (±1 RMSE na perda) para as células com peso de dados zero.
ext = []
for e, r in dec.iterrows():
    perda = r['perda_total_%'] / 100
    sem_dado = {5: 1.0, 8: 746 / 978}.get(e, 0.0)           # parcela do quadro sem dado próprio (notebook 03)
    for nome, d in (('-1 RMSE', -.054), ('base', 0), ('+1 RMSE', .054)):
        p = perda + sem_dado * d
        ext.append({'empresa': e, 'cenário': nome, 'quadro extra %': p / (1 - p) * 100})
ext = pd.DataFrame(ext).pivot(index='empresa', columns='cenário', values='quadro extra %')[['-1 RMSE', 'base', '+1 RMSE']]
ext.round(2).to_csv('resultados/trilha2_faixa_extrapolacao.csv', encoding='utf-8-sig')
print(ext.round(1), end="\n\n")

# Na empresa 5 a faixa honesta é de ~14% a ~30% das posições, bem mais larga que a faixa baixo–alto do
# notebook 03 (19% a 24%), que só refletia a escolha da referência. **É o maior risco do preço, e só se
# reduz com o ponto da empresa 5.**

# ## 5. T3 · A escala praticada está associada à rotatividade?
print(efeito_escala_rotatividade("dados"), end="\n\n")

# Controlando empresa, grupo de função e tempo de casa, o total de saídas não difere entre 6x1 e 5x2, mas as
# **saídas voluntárias são 1,74 vez mais frequentes no 6x1** (IC95% 1,00 a 3,04; p = 0,052). É associação,
# não efeito causal; aponta que o 5x2 pode reduzir as saídas voluntárias (~43% a menos se o excesso
# desaparecer), o que na empresa 6 equivale a ~20% a menos de turnover.
