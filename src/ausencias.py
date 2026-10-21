"""Funções da Trilha 2 (folguistas e feristas).

Ideia central: a "perda de carga" é a fração do trabalho programado que deixa de ser
entregue por ausência. Para entregar N posições, a folha precisa de N / (1 - perda) pessoas,
logo o quadro extra é N * perda / (1 - perda).

A perda é separada em três blocos, que se tratam de formas diferentes:
  ferias      planejada; só quem tem >= 1 ano de casa tira férias
  prolongada  INSS, acidente, licença, aposentadoria por incapacidade (estado persistente)
  curta       faltas, atestados de até 15 dias e abonos (imprevista, varia dia a dia)
"""
import numpy as np
import pandas as pd

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
    from rotatividade import FAIXAS_TEMPO_CASA, envelhecer
    faixa = np.searchsorted(FAIXAS_TEMPO_CASA[1:], np.arange(200), side='left')
    h = risco_faixa[faixa]
    n = np.bincount(np.minimum(np.asarray(tempo_meses, dtype=int), 199), minlength=200).astype(float)
    saida = []
    for _ in range(meses):
        saida.append(n[12:].sum() / n.sum())
        sai = n * h
        n = envelhecer(n - sai, sai.sum())
    return np.array(saida)
