"""Geração de gráficos de distribuição (pizza, histogramas e curvas KDE)."""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.audit_cleaning import ORDEM_CATEGORIAS


def gerar_graficos_distribuicao(
    df: pd.DataFrame, pasta_figuras: str = "figures"
) -> None:
    """Gera gráficos de pizza e subplots verticais (Histograma + KDE) a 300 DPI."""
    os.makedirs(pasta_figuras, exist_ok=True)

    # Gráficos de Pizza
    graficos_pizza = [
        ("Major_Category", "distribuicao_major_category.png", "Distribuição por Major Category",
         ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]),
        ("Burnout_Risk_Level", "distribuicao_burnout_risk.png", "Distribuição por Risco de Burnout",
         ["#4caf50", "#ff9800", "#f44336"]),
    ]
    for col, arq, titulo, cores in graficos_pizza:
        contagem = df[col].value_counts()
        ordem = ORDEM_CATEGORIAS.get(col, sorted(contagem.index))

        valores = []
        for cat in ordem:
            valores.append(contagem[cat])

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(
            valores,
            labels=ordem,
            autopct="%1.1f%%",
            startangle=140,
            colors=cores,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_figuras, arq), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Histograma (topo) e Densidade KDE (base)
    variaveis = [
        ("Pre_Semester_GPA", "Pre-Semester GPA (Média Inicial)", "GPA Inicial", "vertical_pre_gpa.png", "#1f77b4", False),
        ("Post_Semester_GPA", "Post-Semester GPA (Média Final)", "GPA Final", "vertical_post_gpa.png", "#2ca02c", False),
        ("GPA_Variation", "Variação do GPA (Final - Inicial)", "Variação do GPA", "vertical_gpa_variation.png", "#17becf", False),
        ("Weekly_GenAI_Hours", "Horas Semanais de Uso de IA", "Horas Semanais de IA", "vertical_weekly_ai.png", "#3498db", False),
        ("Anxiety_Level_During_Exams", "Nível de Ansiedade em Exames", "Ansiedade (Escala 1 a 10)", "vertical_anxiety.png", "#e74c3c", True),
        ("Traditional_Study_Hours", "Horas Semanais de Estudo Tradicional", "Horas Tradicionais", "vertical_traditional_study.png", "#9b59b6", False),
        ("Perceived_AI_Dependency", "Dependência Percebida de IA", "Dependência (Escala 1 a 10)", "vertical_ai_dependency.png", "#f39c12", True),
        ("Skill_Retention_Score", "Pontuação de Retenção de Habilidades", "Score (0 a 100)", "vertical_skill_retention.png", "#16a085", False),
        ("Tool_Diversity", "Diversidade de Ferramentas de IA", "Quantidade de Ferramentas (1 a 5)", "vertical_tool_diversity.png", "#7f7f7f", True),
    ]

    for col, nome, xlabel, arq, cor, discreta in variaveis:
        serie = df[col].dropna()
        media, mediana = float(serie.mean()), float(serie.median())

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

        # Painel Superior: Histograma
        if discreta:
            minimo, maximo = serie.min(), serie.max()
            bins = np.linspace(minimo - 0.5, maximo + 0.5, int(maximo - minimo) + 2)
        else:
            bins = 30

        axes[0].hist(serie, bins=bins, color=cor, edgecolor="black", alpha=0.7)
        axes[0].axvline(media, color="red", linestyle="--", linewidth=2, label=f"Média: {media:.2f}")
        axes[0].axvline(mediana, color="darkorange", linestyle="-", linewidth=2, label=f"Mediana: {mediana:.2f}")
        axes[0].set_title(f"Histograma: {nome}", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Contagem")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Painel Inferior: Curva KDE
        bw = 2.0 if discreta else 1.0
        sns.kdeplot(serie, color=cor, fill=True, alpha=0.3, linewidth=2, bw_adjust=bw, ax=axes[1])
        axes[1].axvline(media, color="red", linestyle="--", linewidth=2, label=f"Média: {media:.2f}")
        axes[1].axvline(mediana, color="darkorange", linestyle="-", linewidth=2, label=f"Mediana: {mediana:.2f}")
        axes[1].set_title(f"Densidade (KDE): {nome}", fontsize=11, fontweight="bold")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel("Densidade de Probabilidade")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(pasta_figuras, arq), dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Visualizações geradas com sucesso em '{pasta_figuras}/'.")
