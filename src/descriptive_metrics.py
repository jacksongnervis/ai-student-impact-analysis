"""Cálculo e exportação de métricas descritivas e frequências categóricas."""

from __future__ import annotations

import os
import pandas as pd


def calcular_metricas_quantitativas(
    df: pd.DataFrame, pasta_saida: str = "data/processed"
) -> pd.DataFrame:
    """Calcula estatísticas descritivas das variáveis numéricas e exporta CSV."""
    cols = [
        "Pre_Semester_GPA",
        "Post_Semester_GPA",
        "GPA_Variation",
        "Weekly_GenAI_Hours",
        "Traditional_Study_Hours",
        "Perceived_AI_Dependency",
        "Anxiety_Level_During_Exams",
        "Skill_Retention_Score",
        "Tool_Diversity",
    ]

    registros = []
    for col in cols:
        s = df[col].dropna()
        registros.append(
            {
                "Variavel": col,
                "Contagem": int(s.count()),
                "Media": round(float(s.mean()), 2),
                "Mediana": round(float(s.median()), 2),
                "Desvio_Padrao": round(float(s.std()), 2),
                "Minimo": round(float(s.min()), 2),
                "Q1_25": round(float(s.quantile(0.25)), 2),
                "Q3_75": round(float(s.quantile(0.75)), 2),
                "Maximo": round(float(s.max()), 2),
            }
        )

    df_metricas = pd.DataFrame(registros)
    os.makedirs(pasta_saida, exist_ok=True)
    caminho = os.path.join(pasta_saida, "metricas_descritivas_numericas.csv")
    df_metricas.to_csv(caminho, index=False)

    print("\n--- Métricas Descritivas Numéricas ---")
    print(df_metricas.to_string(index=False))
    return df_metricas


def calcular_frequencias_qualitativas(
    df: pd.DataFrame, pasta_saida: str = "data/processed"
) -> pd.DataFrame:
    """Calcula distribuição de frequências absolutas e relativas (%) de variáveis categóricas."""
    cols = [
        "Major_Category",
        "Year_of_Study",
        "Primary_Use_Case",
        "Prompt_Engineering_Skill",
        "Paid_Subscription",
        "Institutional_Policy",
        "Burnout_Risk_Level",
    ]

    registros = []
    for col in cols:
        cont = df[col].value_counts(sort=False)
        pct = (df[col].value_counts(normalize=True, sort=False) * 100).round(2)
        for cat in cont.index:
            registros.append(
                {
                    "Variavel": col,
                    "Categoria": str(cat),
                    "Frequencia_Absoluta": int(cont[cat]),
                    "Frequencia_Relativa_Pct": float(pct[cat]),
                }
            )

    df_freq = pd.DataFrame(registros)
    os.makedirs(pasta_saida, exist_ok=True)
    caminho = os.path.join(pasta_saida, "frequencias_variaveis_categoricas.csv")
    df_freq.to_csv(caminho, index=False)

    print("\n--- Frequências de Variáveis Categóricas ---")
    print(df_freq.to_string(index=False))
    return df_freq
