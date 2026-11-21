"""Análise de duas variáveis: correlação, regressão linear simples, médias por grupo e qui-quadrado."""

import math
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.audit_cleaning import ORDEM_CATEGORIAS, ORDEM_FAIXAS

VARIAVEIS_QUANTITATIVAS_CORRELACAO = [
    "Weekly_GenAI_Hours",
    "Perceived_AI_Dependency",
    "Traditional_Study_Hours",
    "Skill_Retention_Score",
    "Anxiety_Level_During_Exams",
    "Burnout_Risk_Level_Ord",
    "Pre_Semester_GPA",
    "Post_Semester_GPA",
    "GPA_Variation",
    "Tool_Diversity",
]

PARES_REGRESSAO_LINEAR = [
    ("Weekly_GenAI_Hours", "GPA_Variation"),
    ("Weekly_GenAI_Hours", "Post_Semester_GPA"),
    ("Pre_Semester_GPA", "Post_Semester_GPA"),
    ("Weekly_GenAI_Hours", "Traditional_Study_Hours"),
    ("Weekly_GenAI_Hours", "Skill_Retention_Score"),
    ("Perceived_AI_Dependency", "Skill_Retention_Score"),
    ("Weekly_GenAI_Hours", "Anxiety_Level_During_Exams"),
    ("Perceived_AI_Dependency", "Anxiety_Level_During_Exams"),
    ("Weekly_GenAI_Hours", "Perceived_AI_Dependency"),
]

VARIAVEIS_MEDIA_GRUPO = [
    "GPA_Variation",
    "Post_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Traditional_Study_Hours",
    "Skill_Retention_Score",
    "Anxiety_Level_During_Exams",
    "Perceived_AI_Dependency",
]

VARIAVEIS_CRUZADAS_BURNOUT_CONTINGENCIA = [
    "Faixa_Horas_IA",
    "Year_of_Study",
    "Paid_Subscription",
    "Institutional_Policy",
    "Major_Category",
    "Primary_Use_Case",
    "Prompt_Engineering_Skill",
]


def ordemDe(df, col):
    """Devolve a ordem lógica das categorias de uma coluna (ou alfabética, se não houver)."""
    if col == "Faixa_Horas_IA":
        return ORDEM_FAIXAS

    if col in ORDEM_CATEGORIAS:
        return ORDEM_CATEGORIAS[col]

    return sorted(df[col].unique())


def calcularCorrelacoes(df, pastaSaida):
    """Matriz de correlação linear de Pearson (np.corrcoef) entre as variáveis quantitativas."""
    valores = df[VARIAVEIS_QUANTITATIVAS_CORRELACAO].to_numpy()
    matriz = np.corrcoef(valores.T)

    dfCorr = pd.DataFrame(
        matriz,
        index=VARIAVEIS_QUANTITATIVAS_CORRELACAO,
        columns=VARIAVEIS_QUANTITATIVAS_CORRELACAO,
    ).round(3)
    dfCorr.to_csv(os.path.join(pastaSaida, "correlacoes_pearson.csv"))

    print("\n--- Matriz de Correlação de Pearson (r) ---")
    print(dfCorr.to_string())

    return dfCorr


def calcularRegressoes(df, pastaSaida):
    """Ajusta y = b0 + b1*x para cada par e registra b0, b1, r e R²."""
    registros = []

    for xCol, yCol in PARES_REGRESSAO_LINEAR:
        X = df[xCol].to_numpy().reshape(-1, 1)
        Y = df[yCol].to_numpy()

        modelo = LinearRegression()
        modelo.fit(X, Y)

        r = np.corrcoef(df[xCol], df[yCol])[0, 1]

        registros.append({
            "X": xCol,
            "Y": yCol,
            "b0_intercepto": round(float(modelo.intercept_), 4),
            "b1_inclinacao": round(float(modelo.coef_[0]), 4),
            "r_Pearson": round(float(r), 3),
            "R2": round(float(modelo.score(X, Y)), 4),
        })

    dfReg = pd.DataFrame(registros)
    dfReg.to_csv(os.path.join(pastaSaida, "regressoes_lineares_simples.csv"), index=False)

    print("\n--- Regressões Lineares Simples (y = b0 + b1*x) ---")
    print(dfReg.to_string(index=False))

    return dfReg


def calcularMediasPorGrupo(df, colGrupo, pastaSaida):
    """Média das variáveis principais dentro de cada categoria de colGrupo."""
    ordem = ordemDe(df, colGrupo)

    medias = df.groupby(colGrupo)[VARIAVEIS_MEDIA_GRUPO].mean()
    medias = medias.loc[ordem].round(3)

    quantidade = df[colGrupo].value_counts().loc[ordem]
    medias.insert(0, "n", quantidade)

    medias.to_csv(os.path.join(pastaSaida, f"medias_por_{colGrupo}.csv"))

    print(f"\n--- Médias por {colGrupo} ---")
    print(medias.to_string())

    return medias


def calcularContingencia(df, colLinha, pastaSaida, colColuna="Burnout_Risk_Level"):
    """Tabela de contingência, qui-quadrado de Pearson e coeficiente de contingência C."""
    observado = pd.crosstab(df[colLinha], df[colColuna])
    observado = observado.loc[ordemDe(df, colLinha), ordemDe(df, colColuna)]
    n = observado.to_numpy().sum()

    totalLinhas = observado.sum(axis=1).to_numpy().reshape(-1, 1)
    totalColunas = observado.sum(axis=0).to_numpy()
    esperado = totalLinhas * totalColunas / n

    diferencas = ((observado.to_numpy() - esperado) ** 2) / esperado
    quiQuadrado = float(diferencas.sum())
    coefContingencia = math.sqrt(quiQuadrado / (quiQuadrado + n))

    percentual = observado.div(observado.sum(axis=1), axis=0) * 100
    percentual = percentual.round(1)
    percentual.to_csv(os.path.join(pastaSaida, f"contingencia_{colColuna}_por_{colLinha}.csv"))

    print(f"\n--- {colColuna} por {colLinha} (% na linha) ---")
    print(percentual.to_string())
    print(f"Qui-quadrado = {quiQuadrado:.1f} | Coeficiente de contingência C = {coefContingencia:.3f}")

    return {
        "Variavel": colLinha,
        "Qui_Quadrado": round(quiQuadrado, 1),
        "Coef_Contingencia_C": round(coefContingencia, 3),
        "percentual": percentual,
    }


def executarAnaliseBivariada(df, pastaSaida="data/processed"):
    """Executa todas as análises de duas variáveis e devolve os resultados para os gráficos."""
    os.makedirs(pastaSaida, exist_ok=True)

    df = df.copy()
    df["Paid_Subscription"] = df["Paid_Subscription"].astype(str)

    correlacoes = calcularCorrelacoes(df, pastaSaida)
    regressoes = calcularRegressoes(df, pastaSaida)

    colunasMedias = ["Faixa_Horas_IA", "Burnout_Risk_Level"] + VARIAVEIS_CRUZADAS_BURNOUT_CONTINGENCIA[1:]
    medias = {}
    for col in colunasMedias:
        medias[col] = calcularMediasPorGrupo(df, col, pastaSaida)

    contingencias = {}
    for col in VARIAVEIS_CRUZADAS_BURNOUT_CONTINGENCIA:
        contingencias[col] = calcularContingencia(df, col, pastaSaida)

    linhasResumo = []
    for resultado in contingencias.values():
        linhasResumo.append({
            "Variavel": resultado["Variavel"],
            "Qui_Quadrado": resultado["Qui_Quadrado"],
            "Coef_Contingencia_C": resultado["Coef_Contingencia_C"],
        })

    resumo = pd.DataFrame(linhasResumo).sort_values(by="Coef_Contingencia_C", ascending=False)
    resumo.to_csv(os.path.join(pastaSaida, "qui_quadrado_resumo.csv"), index=False)

    print("\n--- Resumo: associação com Burnout_Risk_Level ---")
    print(resumo.to_string(index=False))

    return {
        "correlacoes": correlacoes,
        "regressoes": regressoes,
        "medias": medias,
        "contingencias": contingencias,
        "resumo_qui_quadrado": resumo,
    }
