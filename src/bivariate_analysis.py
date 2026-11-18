"""Análise de duas variáveis: correlação, regressão linear simples, médias por grupo e qui-quadrado."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.audit_cleaning import ORDEM_CATEGORIAS, ORDEM_FAIXAS

# Variáveis quantitativas usadas na matriz de correlação
VARIAVEIS_CORRELACAO = [
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

# Pares (x, y) das regressões lineares simples: H1 nos três primeiros, H2 nos demais
PARES_REGRESSAO: List[Tuple[str, str]] = [
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

# Variáveis cuja média é calculada dentro de cada grupo
VARIAVEIS_MEDIA = [
    "GPA_Variation",
    "Post_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Traditional_Study_Hours",
    "Skill_Retention_Score",
    "Anxiety_Level_During_Exams",
    "Perceived_AI_Dependency",
]

# Variáveis cruzadas com Burnout_Risk_Level nas tabelas de contingência
VARIAVEIS_CONTINGENCIA = [
    "Faixa_Horas_IA",
    "Year_of_Study",
    "Paid_Subscription",
    "Institutional_Policy",
    "Major_Category",
    "Primary_Use_Case",
    "Prompt_Engineering_Skill",
]


def ordem_de(df: pd.DataFrame, col: str) -> list:
    """Devolve a ordem lógica das categorias de uma coluna (ou alfabética, se não houver)."""
    ordens = {**ORDEM_CATEGORIAS, "Faixa_Horas_IA": ORDEM_FAIXAS}
    return ordens.get(col, sorted(df[col].unique()))


def calcular_correlacoes(df: pd.DataFrame, pasta_saida: str) -> pd.DataFrame:
    """Matriz de correlação linear de Pearson (np.corrcoef) entre as variáveis quantitativas."""
    matriz = np.corrcoef(df[VARIAVEIS_CORRELACAO].to_numpy().T)
    df_corr = pd.DataFrame(matriz, index=VARIAVEIS_CORRELACAO, columns=VARIAVEIS_CORRELACAO).round(3)
    df_corr.to_csv(os.path.join(pasta_saida, "correlacoes_pearson.csv"))

    print("\n--- Matriz de Correlação de Pearson (r) ---")
    print(df_corr.to_string())
    return df_corr


def calcular_regressoes(df: pd.DataFrame, pasta_saida: str) -> pd.DataFrame:
    """Ajusta y = b0 + b1*x para cada par e registra b0, b1, r e R²."""
    registros = []
    for x_col, y_col in PARES_REGRESSAO:
        X = df[x_col].to_numpy().reshape(-1, 1)
        Y = df[y_col].to_numpy()
        modelo = LinearRegression()
        modelo.fit(X, Y)
        registros.append(
            {
                "X": x_col,
                "Y": y_col,
                "b0_intercepto": round(float(modelo.intercept_), 4),
                "b1_inclinacao": round(float(modelo.coef_[0]), 4),
                "r_Pearson": round(float(np.corrcoef(df[x_col], df[y_col])[0, 1]), 3),
                "R2": round(float(modelo.score(X, Y)), 4),
            }
        )

    df_reg = pd.DataFrame(registros)
    df_reg.to_csv(os.path.join(pasta_saida, "regressoes_lineares_simples.csv"), index=False)

    print("\n--- Regressões Lineares Simples (y = b0 + b1*x) ---")
    print(df_reg.to_string(index=False))
    return df_reg


def calcular_medias_por_grupo(df: pd.DataFrame, col_grupo: str, pasta_saida: str) -> pd.DataFrame:
    """Média das variáveis principais dentro de cada categoria de col_grupo."""
    ordem = ordem_de(df, col_grupo)
    medias = df.groupby(col_grupo)[VARIAVEIS_MEDIA].mean().loc[ordem].round(3)
    medias.insert(0, "n", df[col_grupo].value_counts().loc[ordem])
    medias.to_csv(os.path.join(pasta_saida, f"medias_por_{col_grupo}.csv"))

    print(f"\n--- Médias por {col_grupo} ---")
    print(medias.to_string())
    return medias


def calcular_contingencia(
    df: pd.DataFrame, col_linha: str, pasta_saida: str, col_coluna: str = "Burnout_Risk_Level"
) -> Dict[str, Any]:
    """Tabela de contingência, qui-quadrado de Pearson e coeficiente de contingência C."""
    observado = pd.crosstab(df[col_linha], df[col_coluna]).loc[ordem_de(df, col_linha), ordem_de(df, col_coluna)]
    n = observado.to_numpy().sum()

    total_linhas = observado.sum(axis=1).to_numpy().reshape(-1, 1)
    total_colunas = observado.sum(axis=0).to_numpy()
    esperado = total_linhas * total_colunas / n

    qui_quadrado = float((((observado.to_numpy() - esperado) ** 2) / esperado).sum())
    coef_contingencia = (qui_quadrado / (qui_quadrado + n)) ** 0.5

    percentual = (observado.div(observado.sum(axis=1), axis=0) * 100).round(1)
    percentual.to_csv(os.path.join(pasta_saida, f"contingencia_{col_coluna}_por_{col_linha}.csv"))

    print(f"\n--- {col_coluna} por {col_linha} (% na linha) ---")
    print(percentual.to_string())
    print(f"Qui-quadrado = {qui_quadrado:.1f} | Coeficiente de contingência C = {coef_contingencia:.3f}")

    return {
        "Variavel": col_linha,
        "Qui_Quadrado": round(qui_quadrado, 1),
        "Coef_Contingencia_C": round(coef_contingencia, 3),
        "percentual": percentual,
    }


def executar_analise_bivariada(df: pd.DataFrame, pasta_saida: str = "data/processed") -> Dict[str, Any]:
    """Executa todas as análises de duas variáveis e devolve os resultados para os gráficos."""
    os.makedirs(pasta_saida, exist_ok=True)
    df = df.copy()
    df["Paid_Subscription"] = df["Paid_Subscription"].astype(str)

    correlacoes = calcular_correlacoes(df, pasta_saida)
    regressoes = calcular_regressoes(df, pasta_saida)

    medias = {}
    for col in ["Faixa_Horas_IA", "Burnout_Risk_Level"] + VARIAVEIS_CONTINGENCIA[1:]:
        medias[col] = calcular_medias_por_grupo(df, col, pasta_saida)

    contingencias = {}
    for col in VARIAVEIS_CONTINGENCIA:
        contingencias[col] = calcular_contingencia(df, col, pasta_saida)

    resumo = pd.DataFrame(
        [{k: v for k, v in c.items() if k != "percentual"} for c in contingencias.values()]
    ).sort_values(by="Coef_Contingencia_C", ascending=False)
    resumo.to_csv(os.path.join(pasta_saida, "qui_quadrado_resumo.csv"), index=False)

    print("\n--- Resumo: associação com Burnout_Risk_Level ---")
    print(resumo.to_string(index=False))

    return {
        "correlacoes": correlacoes,
        "regressoes": regressoes,
        "medias": medias,
        "contingencias": contingencias,
        "resumo_qui_quadrado": resumo,
    }
