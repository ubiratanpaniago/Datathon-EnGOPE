"""Trilha 2: reserva para ausência curta escolhida pelo menor custo esperado.

Num dia com taxa de ausência p, escalam-se Q titulares e R reservas, e cada um falta com probabilidade p
(a reserva também falta). Postos descobertos: U = max(0, Q - S), S ~ Binomial(Q + R, 1 - p).
Custo diário em pessoa-dia: R + k E[U], com k o custo de um posto descoberto.
"""
import numpy as np
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
