"""Aprendizado de máquina: Random Forest e KNN (classificação do burnout) e K-means (perfis de uso)."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.audit_cleaning import ORDEM_CATEGORIAS

SEMENTE = 42
CLASSES_BURNOUT = ORDEM_CATEGORIAS["Burnout_Risk_Level"]

# Atributos numéricos e ordinais (usados pelo Random Forest e pelo KNN)
VARIAVEIS_NUMERICAS = [
    "Weekly_GenAI_Hours",
    "Perceived_AI_Dependency",
    "Traditional_Study_Hours",
    "Anxiety_Level_During_Exams",
    "Skill_Retention_Score",
    "Tool_Diversity",
    "Pre_Semester_GPA",
    "Post_Semester_GPA",
    "Year_of_Study_Ord",
    "Prompt_Engineering_Skill_Ord",
    "Paid_Subscription",
]
# Atributos nominais (codificados como inteiros; usados apenas pelo Random Forest)
VARIAVEIS_NOMINAIS = ["Major_Category", "Primary_Use_Case", "Institutional_Policy"]

PROFUNDIDADES_RF = [5, 10, 15, 20, None]
VALORES_K_KNN = [1, 5, 15, 25, 51, 101, 151, 201]

# K-means: variáveis de hábito de uso e bem-estar; número de grupos escolhido pelo método do cotovelo
VARIAVEIS_KMEANS = [
    "Weekly_GenAI_Hours",
    "Perceived_AI_Dependency",
    "Traditional_Study_Hours",
    "Anxiety_Level_During_Exams",
]
K_GRUPOS = 4


def preparar_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Monta a tabela X: numéricas/ordinais + nominais codificadas como inteiros."""
    X = df[VARIAVEIS_NUMERICAS].copy()
    X["Paid_Subscription"] = X["Paid_Subscription"].astype(int)
    for col in VARIAVEIS_NOMINAIS:
        X[col + "_Cod"] = 0
        for codigo, categoria in enumerate(sorted(df[col].unique())):
            X.loc[df[col] == categoria, col + "_Cod"] = codigo
    return X


def dividir_dados(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Treino 70% / teste 30%; o treino é subdividido em ajuste 80% / validação 20%."""
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, random_state=SEMENTE, stratify=y
    )
    X_ajuste, X_valid, y_ajuste, y_valid = train_test_split(
        X_treino, y_treino, test_size=0.2, random_state=SEMENTE, stratify=y_treino
    )
    print(f"\nTreino: {len(X_treino)} (ajuste {len(X_ajuste)} + validação {len(X_valid)}) | Teste: {len(X_teste)}")
    return {
        "X_treino": X_treino, "y_treino": y_treino,
        "X_teste": X_teste, "y_teste": y_teste,
        "X_ajuste": X_ajuste, "y_ajuste": y_ajuste,
        "X_valid": X_valid, "y_valid": y_valid,
    }


def padronizar(X_referencia: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Escore Z: (x - média) / desvio-padrão, com média e desvio da base de referência."""
    return (X - X_referencia.mean()) / X_referencia.std()


def avaliar_classificador(
    nome: str, y_real: pd.Series, y_previsto: np.ndarray
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Matriz de confusão e métricas por classe (precisão, recall, F1) e médias macro."""
    real = y_real.to_numpy()
    linhas = []
    for classe_real in CLASSES_BURNOUT:
        linhas.append([int(((real == classe_real) & (y_previsto == classe_prev)).sum()) for classe_prev in CLASSES_BURNOUT])
    matriz = pd.DataFrame(
        linhas,
        index=[f"Real_{c}" for c in CLASSES_BURNOUT],
        columns=[f"Previsto_{c}" for c in CLASSES_BURNOUT],
    )

    precisoes, recalls, f1s = [], [], []
    for i in range(len(CLASSES_BURNOUT)):
        tp = matriz.iloc[i, i]
        fp = matriz.iloc[:, i].sum() - tp
        fn = matriz.iloc[i, :].sum() - tp
        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) > 0 else 0.0
        precisoes.append(precisao)
        recalls.append(recall)
        f1s.append(f1)

    acertos = sum(matriz.iloc[i, i] for i in range(len(CLASSES_BURNOUT)))
    metricas = {
        "Modelo": nome,
        "Acuracia": round(float(acertos / matriz.to_numpy().sum()), 4),
        "Precisao_Macro": round(float(np.average(precisoes)), 4),
        "Recall_Macro": round(float(np.average(recalls)), 4),
        "F1_Macro": round(float(np.average(f1s)), 4),
    }
    for classe, f1 in zip(CLASSES_BURNOUT, f1s):
        metricas[f"F1_{classe}"] = round(float(f1), 4)

    print(f"\n--- {nome}: matriz de confusão (teste) ---")
    print(matriz.to_string())
    print(classification_report(real, y_previsto, labels=CLASSES_BURNOUT, digits=3, zero_division=0))
    return metricas, matriz


def treinar_random_forest(dados: Dict[str, Any], pasta_saida: str) -> Dict[str, Any]:
    """Escolhe max_depth na validação, treina no treino completo e avalia no teste."""
    validacao = []
    melhor_profundidade, melhor_acuracia = None, -1.0
    for profundidade in PROFUNDIDADES_RF:
        modelo = RandomForestClassifier(n_estimators=100, max_depth=profundidade, random_state=SEMENTE)
        modelo.fit(dados["X_ajuste"], dados["y_ajuste"])
        acc_ajuste = accuracy_score(dados["y_ajuste"], modelo.predict(dados["X_ajuste"]))
        acc_valid = accuracy_score(dados["y_valid"], modelo.predict(dados["X_valid"]))
        validacao.append(
            {"max_depth": str(profundidade), "Acuracia_Ajuste": round(acc_ajuste, 4), "Acuracia_Validacao": round(acc_valid, 4)}
        )
        if acc_valid > melhor_acuracia:
            melhor_profundidade, melhor_acuracia = profundidade, acc_valid

    df_validacao = pd.DataFrame(validacao)
    df_validacao.to_csv(os.path.join(pasta_saida, "rf_validacao_profundidade.csv"), index=False)
    print("\n--- Random Forest: escolha de max_depth ---")
    print(df_validacao.to_string(index=False))
    print(f"max_depth escolhido: {melhor_profundidade}")

    modelo = RandomForestClassifier(n_estimators=100, max_depth=melhor_profundidade, random_state=SEMENTE)
    modelo.fit(dados["X_treino"], dados["y_treino"])
    previsto = modelo.predict(dados["X_teste"])
    metricas, matriz = avaliar_classificador(f"Random Forest (max_depth={melhor_profundidade})", dados["y_teste"], previsto)
    matriz.to_csv(os.path.join(pasta_saida, "rf_matriz_confusao.csv"))

    importancias = pd.DataFrame(
        {"Variavel": dados["X_treino"].columns, "Importancia": modelo.feature_importances_.round(4)}
    ).sort_values(by="Importancia", ascending=False)
    importancias.to_csv(os.path.join(pasta_saida, "rf_importancia_variaveis.csv"), index=False)
    print("--- Random Forest: importância das variáveis ---")
    print(importancias.to_string(index=False))

    return {"metricas": metricas, "matriz": matriz, "validacao": df_validacao, "importancias": importancias}


def treinar_knn(dados: Dict[str, Any], pasta_saida: str) -> Dict[str, Any]:
    """Escolhe K na validação (atributos padronizados), treina no treino completo e avalia no teste."""
    X_ajuste = padronizar(dados["X_ajuste"][VARIAVEIS_NUMERICAS], dados["X_ajuste"][VARIAVEIS_NUMERICAS])
    X_valid = padronizar(dados["X_ajuste"][VARIAVEIS_NUMERICAS], dados["X_valid"][VARIAVEIS_NUMERICAS])

    validacao = []
    melhor_k, melhor_acuracia = None, -1.0
    for k in VALORES_K_KNN:
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_ajuste, dados["y_ajuste"])
        acc_valid = accuracy_score(dados["y_valid"], modelo.predict(X_valid))
        validacao.append({"K": k, "Acuracia_Validacao": round(acc_valid, 4)})
        if acc_valid > melhor_acuracia:
            melhor_k, melhor_acuracia = k, acc_valid

    df_validacao = pd.DataFrame(validacao)
    df_validacao.to_csv(os.path.join(pasta_saida, "knn_validacao_k.csv"), index=False)
    print("\n--- KNN: escolha de K ---")
    print(df_validacao.to_string(index=False))
    print(f"K escolhido: {melhor_k}")

    X_treino = padronizar(dados["X_treino"][VARIAVEIS_NUMERICAS], dados["X_treino"][VARIAVEIS_NUMERICAS])
    X_teste = padronizar(dados["X_treino"][VARIAVEIS_NUMERICAS], dados["X_teste"][VARIAVEIS_NUMERICAS])
    modelo = KNeighborsClassifier(n_neighbors=melhor_k)
    modelo.fit(X_treino, dados["y_treino"])
    previsto = modelo.predict(X_teste)
    metricas, matriz = avaliar_classificador(f"KNN (K={melhor_k})", dados["y_teste"], previsto)
    matriz.to_csv(os.path.join(pasta_saida, "knn_matriz_confusao.csv"))

    return {"metricas": metricas, "matriz": matriz, "validacao": df_validacao}


def agrupar_kmeans(df: pd.DataFrame, pasta_saida: str) -> Dict[str, Any]:
    """Método do cotovelo (k = 1 a 10) e perfil dos K_GRUPOS grupos formados pelo K-means."""
    Z = padronizar(df[VARIAVEIS_KMEANS], df[VARIAVEIS_KMEANS])

    inercias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=SEMENTE)
        kmeans.fit(Z)
        inercias.append({"k": k, "Inercia": round(float(kmeans.inertia_), 1)})
    df_inercias = pd.DataFrame(inercias)
    df_inercias.to_csv(os.path.join(pasta_saida, "kmeans_cotovelo.csv"), index=False)
    print("\n--- K-means: método do cotovelo ---")
    print(df_inercias.to_string(index=False))

    kmeans = KMeans(n_clusters=K_GRUPOS, random_state=SEMENTE)
    kmeans.fit(Z)
    df_grupos = df.copy()
    df_grupos["Grupo_Original"] = kmeans.predict(Z)

    # Renumera os grupos de 1 a K em ordem crescente de horas médias de IA
    ordem = df_grupos.groupby("Grupo_Original")["Weekly_GenAI_Hours"].mean().sort_values().index
    df_grupos["Grupo"] = 0
    for novo, original in enumerate(ordem, start=1):
        df_grupos.loc[df_grupos["Grupo_Original"] == original, "Grupo"] = novo

    numeros = list(range(1, K_GRUPOS + 1))
    perfil = df_grupos.groupby("Grupo")[VARIAVEIS_KMEANS + ["Skill_Retention_Score", "GPA_Variation"]].mean().loc[numeros].round(2)
    perfil.insert(0, "n", df_grupos["Grupo"].value_counts().loc[numeros])
    for nivel in CLASSES_BURNOUT:
        perfil[f"Pct_Burnout_{nivel}"] = (
            (df_grupos["Burnout_Risk_Level"] == nivel).groupby(df_grupos["Grupo"]).mean().loc[numeros] * 100
        ).round(1)
    perfil.to_csv(os.path.join(pasta_saida, "kmeans_perfil_grupos.csv"))
    print(f"\n--- K-means (k={K_GRUPOS}): perfil médio de cada grupo ---")
    print(perfil.to_string())

    return {"inercias": df_inercias, "perfil": perfil, "grupos": df_grupos["Grupo"]}


def executar_modelos(df: pd.DataFrame, pasta_saida: str = "data/processed") -> Dict[str, Any]:
    """Executa a classificação do burnout (baseline, RF, KNN) e o agrupamento K-means."""
    os.makedirs(pasta_saida, exist_ok=True)

    X = preparar_atributos(df)
    y = df["Burnout_Risk_Level"]
    dados = dividir_dados(X, y)

    classe_mais_frequente = dados["y_treino"].value_counts().index[0]
    previsto_base = np.array([classe_mais_frequente] * len(dados["y_teste"]))
    metricas_base, _ = avaliar_classificador(f"Baseline (sempre '{classe_mais_frequente}')", dados["y_teste"], previsto_base)

    rf = treinar_random_forest(dados, pasta_saida)
    knn = treinar_knn(dados, pasta_saida)

    df_metricas = pd.DataFrame([metricas_base, rf["metricas"], knn["metricas"]])
    df_metricas.to_csv(os.path.join(pasta_saida, "modelos_metricas_teste.csv"), index=False)
    print("\n--- Comparação dos classificadores no conjunto de teste ---")
    print(df_metricas.to_string(index=False))

    kmeans = agrupar_kmeans(df, pasta_saida)

    return {"random_forest": rf, "knn": knn, "metricas": df_metricas, "kmeans": kmeans}
