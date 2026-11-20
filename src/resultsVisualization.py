"""Gráficos dos resultados do Milestone 2 (hipóteses H1 e H2, classificadores e K-means)."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.audit_cleaning import ORDEM_CATEGORIAS, ORDEM_FAIXAS

CORES_BURNOUT = {"Low": "#4caf50", "Medium": "#ff9800", "High": "#f44336"}
CORES_GRUPOS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728"]

NOMES = {
    "Weekly_GenAI_Hours": "Horas semanais de IA",
    "Perceived_AI_Dependency": "Dependência percebida de IA (1 a 10)",
    "Traditional_Study_Hours": "Horas semanais de estudo tradicional",
    "Skill_Retention_Score": "Retenção de habilidades (0 a 100)",
    "Anxiety_Level_During_Exams": "Ansiedade em exames (1 a 10)",
    "GPA_Variation": "Variação do GPA (final - inicial)",
    "Pre_Semester_GPA": "GPA inicial",
    "Post_Semester_GPA": "GPA final",
}

ROTULOS_CURTOS = {
    "Weekly_GenAI_Hours": "Horas de IA",
    "Perceived_AI_Dependency": "Dependência",
    "Traditional_Study_Hours": "Estudo tradicional",
    "Skill_Retention_Score": "Retenção",
    "Anxiety_Level_During_Exams": "Ansiedade",
    "Burnout_Risk_Level_Ord": "Burnout (0-2)",
    "Pre_Semester_GPA": "GPA inicial",
    "Post_Semester_GPA": "GPA final",
    "GPA_Variation": "Variação GPA",
    "Tool_Diversity": "Nº ferramentas",
}


def salvar(fig, pasta, arquivo):
    """Ajusta margens, salva a figura a 300 DPI e libera a memória."""
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, arquivo), dpi=300, bbox_inches="tight")
    plt.close(fig)


def dispersaoComReta(ax, df, regressoes, x, y, cor, bolhas=False):
    """Dispersão de x por y com a reta y = b0 + b1*x; com bolhas=True o tamanho do ponto é a quantidade de alunos."""
    filtroPar = (regressoes["X"] == x) & (regressoes["Y"] == y)
    reg = regressoes.loc[filtroPar].iloc[0]

    if bolhas:
        contagem = df.groupby([x, y]).size()

        valoresX = []
        valoresY = []
        for par in contagem.index:
            valoresX.append(par[0])
            valoresY.append(par[1])

        ax.scatter(
            valoresX,
            valoresY,
            s=contagem.to_numpy() / contagem.max() * 400,
            alpha=0.5,
            color=cor,
            edgecolor="black",
        )
    else:
        ax.scatter(df[x], df[y], s=2, alpha=0.1, color=cor)

    b0 = reg["b0_intercepto"]
    b1 = reg["b1_inclinacao"]
    legenda = f"y = {b0:.2f} {b1:+.4f}x | r = {reg['r_Pearson']:.3f} | R² = {reg['R2']:.3f}"

    xs = np.linspace(df[x].min(), df[x].max(), 100)
    ax.plot(xs, b0 + b1 * xs, color="black", linewidth=2, label=legenda)

    ax.set_xlabel(NOMES[x])
    ax.set_ylabel(NOMES[y])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def boxplotPorGrupo(ax, df, colGrupo, ordem, y, cor):
    """Boxplot de y para cada categoria de colGrupo, com a média marcada por um losango."""
    dados = []
    medias = []
    for g in ordem:
        valoresGrupo = df.loc[df[colGrupo] == g, y]
        dados.append(valoresGrupo)
        medias.append(valoresGrupo.mean())

    ax.boxplot(
        dados,
        patch_artist=True,
        boxprops=dict(facecolor=cor, alpha=0.6),
        medianprops=dict(color="black"),
        flierprops=dict(markersize=2, alpha=0.3),
    )

    posicoes = range(1, len(ordem) + 1)
    ax.plot(posicoes, medias, marker="D", color="red", linestyle="--", label="Média")

    ax.set_xticks(posicoes)
    ax.set_xticklabels(ordem, rotation=20)
    ax.set_ylabel(NOMES[y])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def barrasAgrupadas(ax, tabela, cores, rotuloY):
    """Barras lado a lado: cada linha da tabela é um grupo no eixo X e cada coluna uma barra."""
    largura = 0.8 / len(tabela.columns)
    posicoes = np.array(range(len(tabela.index)))

    for i, coluna in enumerate(tabela.columns):
        barras = ax.bar(
            posicoes + i * largura,
            tabela[coluna],
            width=largura,
            color=cores[i],
            edgecolor="black",
            label=str(coluna),
        )

        for barra, valor in zip(barras, tabela[coluna]):
            centroBarra = barra.get_x() + barra.get_width() / 2
            ax.text(centroBarra, valor, f"{valor:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(posicoes + largura * (len(tabela.columns) - 1) / 2)

    nomesGrupos = []
    for i in tabela.index:
        nomesGrupos.append(str(i))
    ax.set_xticklabels(nomesGrupos)

    ax.set_ylabel(rotuloY)
    ax.grid(True, axis="y", alpha=0.3)


def graficosH1(df, bivariada, pasta):
    """H1: horas de IA e variação do GPA."""
    regressoes = bivariada["regressoes"]

    # 1. Dispersão horas x variação do GPA
    fig, ax = plt.subplots(figsize=(9, 6))

    dispersaoComReta(ax, df, regressoes, "Weekly_GenAI_Hours", "GPA_Variation", "#1f77b4")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_title("H1 - Horas semanais de IA x Variação do GPA", fontweight="bold")

    salvar(fig, pasta, "m2_h1_dispersao_horas_variacao_gpa.png")

    # 2. Boxplot da variação do GPA por faixa de horas
    fig, ax = plt.subplots(figsize=(9, 6))

    boxplotPorGrupo(ax, df, "Faixa_Horas_IA", ORDEM_FAIXAS, "GPA_Variation", "#17becf")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Faixa de horas semanais de IA")
    ax.set_title("H1 - Variação do GPA por faixa de horas de IA", fontweight="bold")

    salvar(fig, pasta, "m2_h1_boxplot_variacao_por_faixa.png")

    # 3. GPA inicial x final colorido pelas horas de IA
    fig, ax = plt.subplots(figsize=(8, 7))

    pontos = ax.scatter(
        df["Pre_Semester_GPA"],
        df["Post_Semester_GPA"],
        c=df["Weekly_GenAI_Hours"],
        cmap="viridis",
        s=2,
        alpha=0.4,
    )
    fig.colorbar(pontos, ax=ax, label=NOMES["Weekly_GenAI_Hours"])

    ax.plot([1, 4], [1, 4], "r--", linewidth=2, label="Sem variação (final = inicial)")

    ax.set_xlabel(NOMES["Pre_Semester_GPA"])
    ax.set_ylabel(NOMES["Post_Semester_GPA"])
    ax.set_title("H1 - GPA inicial x GPA final (cor = horas semanais de IA)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    salvar(fig, pasta, "m2_h1_dispersao_pre_pos_por_horas.png")

    # 4. Variação do GPA por finalidade de uso e por habilidade em prompts
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    usos = sorted(df["Primary_Use_Case"].unique())
    boxplotPorGrupo(axes[0], df, "Primary_Use_Case", usos, "GPA_Variation", "#bcbd22")

    nomesUsos = []
    for u in usos:
        nomesUsos.append(u.replace("_", " ").replace("/", "/\n"))
    axes[0].set_xticklabels(nomesUsos, rotation=0, fontsize=8)
    axes[0].set_xlabel("Finalidade principal de uso da IA")
    axes[0].set_title("Por finalidade de uso", fontweight="bold")

    ordemPrompt = ORDEM_CATEGORIAS["Prompt_Engineering_Skill"]
    boxplotPorGrupo(axes[1], df, "Prompt_Engineering_Skill", ordemPrompt, "GPA_Variation", "#e377c2")
    axes[1].set_xlabel("Habilidade em engenharia de prompt")
    axes[1].set_title("Por habilidade em prompts", fontweight="bold")

    fig.suptitle("H1 - Variação do GPA por forma de uso da IA", fontweight="bold")
    salvar(fig, pasta, "m2_h1_boxplot_variacao_por_uso_e_prompt.png")


def graficosH2(df, bivariada, pasta):
    """H2: horas/dependência de IA x estudo tradicional, retenção, ansiedade e burnout."""
    regressoes = bivariada["regressoes"]

    # 0. Mapa de calor da matriz de correlação de Pearson
    correlacoes = bivariada["correlacoes"]

    rotulos = []
    for c in correlacoes.columns:
        rotulos.append(ROTULOS_CURTOS[c])

    fig, ax = plt.subplots(figsize=(10, 8))
    imagem = ax.imshow(correlacoes.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)

    for i in range(len(rotulos)):
        for j in range(len(rotulos)):
            valor = correlacoes.iloc[i, j]

            if abs(valor) > 0.6:
                corTexto = "white"
            else:
                corTexto = "black"

            ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=8, color=corTexto)

    ax.set_xticks(range(len(rotulos)))
    ax.set_xticklabels(rotulos, rotation=45, ha="right")
    ax.set_yticks(range(len(rotulos)))
    ax.set_yticklabels(rotulos)

    fig.colorbar(imagem, ax=ax, label="Coeficiente de correlação de Pearson (r)")
    ax.set_title("Matriz de correlação de Pearson entre as variáveis quantitativas", fontweight="bold")

    salvar(fig, pasta, "m2_correlacao_heatmap.png")

    # 5. Quatro dispersões com reta de regressão
    pares = [
        ("Weekly_GenAI_Hours", "Traditional_Study_Hours", "#9b59b6", False),
        ("Weekly_GenAI_Hours", "Skill_Retention_Score", "#16a085", False),
        ("Weekly_GenAI_Hours", "Perceived_AI_Dependency", "#f39c12", False),
        ("Perceived_AI_Dependency", "Anxiety_Level_During_Exams", "#e74c3c", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (x, y, cor, bolhas) in zip(axes.flatten(), pares):
        dispersaoComReta(ax, df, regressoes, x, y, cor, bolhas)
        ax.set_title(f"{NOMES[x]} x {NOMES[y]}", fontsize=10, fontweight="bold")

    tituloBolhas = (
        f"{NOMES['Perceived_AI_Dependency']} x {NOMES['Anxiety_Level_During_Exams']}\n"
        "(tamanho da bolha = quantidade de alunos)"
    )
    axes[1, 1].set_title(tituloBolhas, fontsize=10, fontweight="bold")

    fig.suptitle("H2 - Relações lineares entre uso de IA, estudo, retenção e ansiedade", fontweight="bold")
    salvar(fig, pasta, "m2_h2_dispersoes_regressao.png")

    # 6. Médias por faixa de horas de IA
    medias = bivariada["medias"]["Faixa_Horas_IA"]
    variaveis = [
        ("Traditional_Study_Hours", "#9b59b6"),
        ("Skill_Retention_Score", "#16a085"),
        ("Anxiety_Level_During_Exams", "#e74c3c"),
        ("Perceived_AI_Dependency", "#f39c12"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, (y, cor) in zip(axes.flatten(), variaveis):
        ax.plot(ORDEM_FAIXAS, medias[y], marker="o", color=cor, linewidth=2)

        for faixa, valor in zip(ORDEM_FAIXAS, medias[y]):
            ax.annotate(
                f"{valor:.2f}",
                (faixa, valor),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )

        ax.set_title(f"Média de {NOMES[y]}", fontsize=10, fontweight="bold")
        ax.set_ylabel(NOMES[y])
        ax.set_xlabel("Faixa de horas semanais de IA")
        ax.grid(True, alpha=0.3)

    fig.suptitle("H2 - Médias por faixa de horas semanais de IA", fontweight="bold")
    salvar(fig, pasta, "m2_h2_medias_por_faixa.png")

    # 7. Boxplots por nível de burnout
    niveis = ORDEM_CATEGORIAS["Burnout_Risk_Level"]
    variaveis = [
        ("Weekly_GenAI_Hours", "#3498db"),
        ("Perceived_AI_Dependency", "#f39c12"),
        ("Anxiety_Level_During_Exams", "#e74c3c"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for ax, (y, cor) in zip(axes, variaveis):
        boxplotPorGrupo(ax, df, "Burnout_Risk_Level", niveis, y, cor)
        ax.set_xlabel("Risco de burnout")
        ax.set_title(NOMES[y], fontsize=10, fontweight="bold")

    fig.suptitle("H2 - Uso de IA, dependência e ansiedade por nível de risco de burnout", fontweight="bold")
    salvar(fig, pasta, "m2_h2_boxplot_por_burnout.png")

    # 8. Percentual de cada nível de burnout por faixa de horas
    percentual = bivariada["contingencias"]["Faixa_Horas_IA"]["percentual"]
    c = bivariada["contingencias"]["Faixa_Horas_IA"]["Coef_Contingencia_C"]

    coresBarras = []
    for n in percentual.columns:
        coresBarras.append(CORES_BURNOUT[n])

    fig, ax = plt.subplots(figsize=(10, 6))

    barrasAgrupadas(ax, percentual, coresBarras, "% dos alunos da faixa")
    ax.set_xlabel("Faixa de horas semanais de IA")
    ax.legend(title="Risco de burnout")
    ax.set_title(
        f"H2 - Risco de burnout por faixa de horas de IA (coef. de contingência C = {c:.3f})",
        fontweight="bold",
    )

    salvar(fig, pasta, "m2_h2_burnout_por_faixa.png")

    # 9. Coeficiente de contingência C de cada variável com o burnout
    resumo = bivariada["resumo_qui_quadrado"]

    fig, ax = plt.subplots(figsize=(10, 6))

    barras = ax.bar(resumo["Variavel"], resumo["Coef_Contingencia_C"], color="#7f7f7f", edgecolor="black")
    for barra, valor in zip(barras, resumo["Coef_Contingencia_C"]):
        centroBarra = barra.get_x() + barra.get_width() / 2
        ax.text(centroBarra, valor, f"{valor:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(resumo)))
    ax.set_xticklabels(resumo["Variavel"], rotation=30, ha="right")
    ax.set_ylabel("Coeficiente de contingência C")
    ax.set_title("H2 - Força da associação de cada variável com o risco de burnout", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    salvar(fig, pasta, "m2_h2_associacao_burnout.png")


def graficosModelos(df, modelos, pasta):
    """Random Forest, KNN e K-means."""
    # 10. Overfitting do Random Forest
    valRf = modelos["random_forest"]["validacao"]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(valRf["max_depth"], valRf["Acuracia_Ajuste"], marker="o", linewidth=2, label="Treino (ajuste)")
    ax.plot(valRf["max_depth"], valRf["Acuracia_Validacao"], marker="s", linewidth=2, label="Validação")

    ax.set_xlabel("Profundidade máxima das árvores (max_depth)")
    ax.set_ylabel("Acurácia")
    ax.set_title("Random Forest - Acurácia no treino x validação (overfitting)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    salvar(fig, pasta, "m2_rf_overfitting_profundidade.png")

    # 11. Escolha do K do KNN
    valKnn = modelos["knn"]["validacao"]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(valKnn["K"], valKnn["Acuracia_Validacao"], marker="o", linewidth=2, color="#8c564b")

    ax.set_xlabel("Número de vizinhos (K)")
    ax.set_ylabel("Acurácia na validação")
    ax.set_title("KNN - Acurácia na validação para cada K", fontweight="bold")
    ax.grid(True, alpha=0.3)

    salvar(fig, pasta, "m2_knn_validacao_k.png")

    # 12. Comparação dos classificadores no teste
    metricas = modelos["metricas"].set_index("Modelo")
    metricas = metricas[["Acuracia", "F1_Macro", "F1_High"]] * 100

    fig, ax = plt.subplots(figsize=(11, 6))

    barrasAgrupadas(ax, metricas, ["#1f77b4", "#ff7f0e", "#f44336"], "Valor da métrica (%)")
    ax.legend(["Acurácia", "F1 macro", "F1 da classe High"], loc="upper left")
    ax.set_title("Classificação do risco de burnout - métricas no conjunto de teste", fontweight="bold")

    salvar(fig, pasta, "m2_modelos_comparacao.png")

    # 13. Importância das variáveis no Random Forest
    imp = modelos["random_forest"]["importancias"]
    importanciaPct = imp["Importancia"] * 100

    fig, ax = plt.subplots(figsize=(11, 6))

    barras = ax.bar(imp["Variavel"], importanciaPct, color="#2ca02c", edgecolor="black")
    for barra, valor in zip(barras, importanciaPct):
        centroBarra = barra.get_x() + barra.get_width() / 2
        ax.text(centroBarra, valor, f"{valor:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels(imp["Variavel"], rotation=40, ha="right")
    ax.set_ylabel("Importância (%)")
    ax.set_title("Random Forest - Importância de cada variável na previsão do burnout", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    salvar(fig, pasta, "m2_rf_importancia_variaveis.png")

    # 14. Método do cotovelo
    kmeans = modelos["kmeans"]
    inercias = kmeans["inercias"]
    kEscolhido = len(kmeans["perfil"])
    inerciaEscolhida = inercias.loc[inercias["k"] == kEscolhido, "Inercia"]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(inercias["k"], inercias["Inercia"], marker="o", linewidth=2)
    ax.plot(
        kEscolhido,
        inerciaEscolhida,
        marker="*",
        markersize=20,
        color="red",
        label=f"k escolhido = {kEscolhido}",
    )

    ax.set_xlabel("Número de grupos (k)")
    ax.set_ylabel("Inércia")
    ax.set_title("K-means - Método do cotovelo", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    salvar(fig, pasta, "m2_kmeans_cotovelo.png")

    # 15. Grupos do K-means: dispersão horas x estudo tradicional e boxplots de ansiedade e dependência
    grupos = kmeans["grupos"]
    perfil = kmeans["perfil"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for numero, cor in zip(perfil.index, CORES_GRUPOS):
        filtro = grupos == numero

        axes[0].scatter(
            df.loc[filtro, "Weekly_GenAI_Hours"],
            df.loc[filtro, "Traditional_Study_Hours"],
            s=2,
            alpha=0.2,
            color=cor,
        )

        axes[0].scatter(
            perfil.loc[numero, "Weekly_GenAI_Hours"],
            perfil.loc[numero, "Traditional_Study_Hours"],
            marker="*",
            s=400,
            color=cor,
            edgecolor="black",
            label=f"Grupo {numero} (n = {perfil.loc[numero, 'n']})",
        )

    axes[0].set_xlabel(NOMES["Weekly_GenAI_Hours"])
    axes[0].set_ylabel(NOMES["Traditional_Study_Hours"])
    axes[0].set_title("Horas de IA x estudo tradicional (estrela = média)", fontsize=10, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    dfGrupos = df.copy()
    dfGrupos["Grupo"] = grupos

    nomesGrupos = []
    for g in perfil.index:
        nomesGrupos.append(f"Grupo {g}")

    variaveis = [
        ("Anxiety_Level_During_Exams", "#e74c3c"),
        ("Perceived_AI_Dependency", "#f39c12"),
    ]
    for ax, (y, cor) in zip(axes[1:], variaveis):
        boxplotPorGrupo(ax, dfGrupos, "Grupo", list(perfil.index), y, cor)
        ax.set_xticklabels(nomesGrupos, rotation=0)
        ax.set_xlabel("Grupo formado pelo K-means")
        ax.set_title(NOMES[y], fontsize=10, fontweight="bold")

    fig.suptitle("K-means - Características dos 4 grupos de alunos", fontweight="bold")
    salvar(fig, pasta, "m2_kmeans_grupos.png")

    # 16. Risco de burnout em cada grupo
    colunasBurnout = []
    for n in ORDEM_CATEGORIAS["Burnout_Risk_Level"]:
        colunasBurnout.append(f"Pct_Burnout_{n}")

    pct = perfil[colunasBurnout]
    pct.columns = ORDEM_CATEGORIAS["Burnout_Risk_Level"]

    nomesGrupos = []
    for g in pct.index:
        nomesGrupos.append(f"Grupo {g}")
    pct.index = nomesGrupos

    coresBarras = []
    for n in pct.columns:
        coresBarras.append(CORES_BURNOUT[n])

    fig, ax = plt.subplots(figsize=(10, 6))

    barrasAgrupadas(ax, pct, coresBarras, "% dos alunos do grupo")
    ax.set_xlabel("Grupo formado pelo K-means")
    ax.legend(title="Risco de burnout")
    ax.set_title("K-means - Risco de burnout em cada grupo", fontweight="bold")

    salvar(fig, pasta, "m2_kmeans_burnout_por_grupo.png")


def gerarGraficosResultados(df, bivariada, modelos, pastaFiguras="figures"):
    """Gera os 17 gráficos do Milestone 2 a 300 DPI."""
    os.makedirs(pastaFiguras, exist_ok=True)

    graficosH1(df, bivariada, pastaFiguras)
    graficosH2(df, bivariada, pastaFiguras)
    graficosModelos(df, modelos, pastaFiguras)

    print(f"Gráficos do Milestone 2 gerados em '{pastaFiguras}/'.")
