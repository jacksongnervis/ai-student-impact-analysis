"""Aprendizado de máquina: Random Forest e KNN (classificação do burnout) e K-means (perfis de uso)."""

import os
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


def prepararAtributos(df):
    """Monta a tabela X: numéricas/ordinais + nominais codificadas como inteiros."""
    X = df[VARIAVEIS_NUMERICAS].copy()
    X["Paid_Subscription"] = X["Paid_Subscription"].astype(int)
    for col in VARIAVEIS_NOMINAIS:
        X[col + "_Cod"] = 0
        for codigo, categoria in enumerate(sorted(df[col].unique())):
            X.loc[df[col] == categoria, col + "_Cod"] = codigo
    return X


def dividirDados(X, y):
    """Treino 70% / teste 30%; o treino é subdividido em ajuste 80% / validação 20%."""
    XTreino, XTeste, yTreino, yTeste = train_test_split(
        X, y, test_size=0.3, random_state=SEMENTE, stratify=y
    )
    XAjuste, XValid, yAjuste, yValid = train_test_split(
        XTreino, yTreino, test_size=0.2, random_state=SEMENTE, stratify=yTreino
    )
    print(f"\nTreino: {len(XTreino)} (ajuste {len(XAjuste)} + validação {len(XValid)}) | Teste: {len(XTeste)}")
    return {
        "X_treino": XTreino, "y_treino": yTreino,
        "X_teste": XTeste, "y_teste": yTeste,
        "X_ajuste": XAjuste, "y_ajuste": yAjuste,
        "X_valid": XValid, "y_valid": yValid,
    }


def padronizar(XReferencia, X):
    """Escore Z: (x - média) / desvio-padrão, com média e desvio da base de referência."""
    return (X - XReferencia.mean()) / XReferencia.std()


def avaliarClassificador(nome, yReal, yPrevisto):
    """Matriz de confusão e métricas por classe (precisão, recall, F1) e médias macro."""
    real = yReal.to_numpy()
    linhas = []
    for classeReal in CLASSES_BURNOUT:
        linhas.append([int(((real == classeReal) & (yPrevisto == classePrev)).sum()) for classePrev in CLASSES_BURNOUT])
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
    print(classification_report(real, yPrevisto, labels=CLASSES_BURNOUT, digits=3, zero_division=0))
    return metricas, matriz


def treinarRandomForest(dados, pastaSaida):
    """Escolhe max_depth na validação, treina no treino completo e avalia no teste."""
    validacao = []
    melhorProfundidade, melhorAcuracia = None, -1.0
    for profundidade in PROFUNDIDADES_RF:
        modelo = RandomForestClassifier(n_estimators=100, max_depth=profundidade, random_state=SEMENTE)
        modelo.fit(dados["X_ajuste"], dados["y_ajuste"])
        accAjuste = accuracy_score(dados["y_ajuste"], modelo.predict(dados["X_ajuste"]))
        accValid = accuracy_score(dados["y_valid"], modelo.predict(dados["X_valid"]))
        validacao.append(
            {"max_depth": str(profundidade), "Acuracia_Ajuste": round(accAjuste, 4), "Acuracia_Validacao": round(accValid, 4)}
        )
        if accValid > melhorAcuracia:
            melhorProfundidade, melhorAcuracia = profundidade, accValid

    dfValidacao = pd.DataFrame(validacao)
    dfValidacao.to_csv(os.path.join(pastaSaida, "rf_validacao_profundidade.csv"), index=False)
    print("\n--- Random Forest: escolha de max_depth ---")
    print(dfValidacao.to_string(index=False))
    print(f"max_depth escolhido: {melhorProfundidade}")

    modelo = RandomForestClassifier(n_estimators=100, max_depth=melhorProfundidade, random_state=SEMENTE)
    modelo.fit(dados["X_treino"], dados["y_treino"])
    previsto = modelo.predict(dados["X_teste"])
    metricas, matriz = avaliarClassificador(f"Random Forest (max_depth={melhorProfundidade})", dados["y_teste"], previsto)
    matriz.to_csv(os.path.join(pastaSaida, "rf_matriz_confusao.csv"))

    importancias = pd.DataFrame(
        {"Variavel": dados["X_treino"].columns, "Importancia": modelo.feature_importances_.round(4)}
    ).sort_values(by="Importancia", ascending=False)
    importancias.to_csv(os.path.join(pastaSaida, "rf_importancia_variaveis.csv"), index=False)
    print("--- Random Forest: importância das variáveis ---")
    print(importancias.to_string(index=False))

    return {"metricas": metricas, "matriz": matriz, "validacao": dfValidacao, "importancias": importancias}


def treinarKnn(dados, pastaSaida):
    """Escolhe K na validação (atributos padronizados), treina no treino completo e avalia no teste."""
    XAjuste = padronizar(dados["X_ajuste"][VARIAVEIS_NUMERICAS], dados["X_ajuste"][VARIAVEIS_NUMERICAS])
    XValid = padronizar(dados["X_ajuste"][VARIAVEIS_NUMERICAS], dados["X_valid"][VARIAVEIS_NUMERICAS])

    validacao = []
    melhorK, melhorAcuracia = None, -1.0
    for k in VALORES_K_KNN:
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(XAjuste, dados["y_ajuste"])
        accValid = accuracy_score(dados["y_valid"], modelo.predict(XValid))
        validacao.append({"K": k, "Acuracia_Validacao": round(accValid, 4)})
        if accValid > melhorAcuracia:
            melhorK, melhorAcuracia = k, accValid

    dfValidacao = pd.DataFrame(validacao)
    dfValidacao.to_csv(os.path.join(pastaSaida, "knn_validacao_k.csv"), index=False)
    print("\n--- KNN: escolha de K ---")
    print(dfValidacao.to_string(index=False))
    print(f"K escolhido: {melhorK}")

    XTreino = padronizar(dados["X_treino"][VARIAVEIS_NUMERICAS], dados["X_treino"][VARIAVEIS_NUMERICAS])
    XTeste = padronizar(dados["X_treino"][VARIAVEIS_NUMERICAS], dados["X_teste"][VARIAVEIS_NUMERICAS])
    modelo = KNeighborsClassifier(n_neighbors=melhorK)
    modelo.fit(XTreino, dados["y_treino"])
    previsto = modelo.predict(XTeste)
    metricas, matriz = avaliarClassificador(f"KNN (K={melhorK})", dados["y_teste"], previsto)
    matriz.to_csv(os.path.join(pastaSaida, "knn_matriz_confusao.csv"))

    return {"metricas": metricas, "matriz": matriz, "validacao": dfValidacao}


def agruparKmeans(df, pastaSaida):
    """Método do cotovelo (k = 1 a 10) e perfil dos K_GRUPOS grupos formados pelo K-means."""
    Z = padronizar(df[VARIAVEIS_KMEANS], df[VARIAVEIS_KMEANS])

    inercias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=SEMENTE)
        kmeans.fit(Z)
        inercias.append({"k": k, "Inercia": round(float(kmeans.inertia_), 1)})
    dfInercias = pd.DataFrame(inercias)
    dfInercias.to_csv(os.path.join(pastaSaida, "kmeans_cotovelo.csv"), index=False)
    print("\n--- K-means: método do cotovelo ---")
    print(dfInercias.to_string(index=False))

    kmeans = KMeans(n_clusters=K_GRUPOS, random_state=SEMENTE)
    kmeans.fit(Z)
    dfGrupos = df.copy()
    dfGrupos["Grupo_Original"] = kmeans.predict(Z)

    # Renumera os grupos de 1 a K em ordem crescente de horas médias de IA
    ordem = dfGrupos.groupby("Grupo_Original")["Weekly_GenAI_Hours"].mean().sort_values().index
    dfGrupos["Grupo"] = 0
    for novo, original in enumerate(ordem, start=1):
        dfGrupos.loc[dfGrupos["Grupo_Original"] == original, "Grupo"] = novo

    numeros = list(range(1, K_GRUPOS + 1))
    perfil = dfGrupos.groupby("Grupo")[VARIAVEIS_KMEANS + ["Skill_Retention_Score", "GPA_Variation"]].mean().loc[numeros].round(2)
    perfil.insert(0, "n", dfGrupos["Grupo"].value_counts().loc[numeros])
    for nivel in CLASSES_BURNOUT:
        perfil[f"Pct_Burnout_{nivel}"] = (
            (dfGrupos["Burnout_Risk_Level"] == nivel).groupby(dfGrupos["Grupo"]).mean().loc[numeros] * 100
        ).round(1)
    perfil.to_csv(os.path.join(pastaSaida, "kmeans_perfil_grupos.csv"))
    print(f"\n--- K-means (k={K_GRUPOS}): perfil médio de cada grupo ---")
    print(perfil.to_string())

    return {"inercias": dfInercias, "perfil": perfil, "grupos": dfGrupos["Grupo"]}


def executarModelos(df, pastaSaida="data/processed"):
    """Executa a classificação do burnout (baseline, RF, KNN) e o agrupamento K-means."""
    os.makedirs(pastaSaida, exist_ok=True)

    X = prepararAtributos(df)
    y = df["Burnout_Risk_Level"]
    dados = dividirDados(X, y)

    classeMaisFrequente = dados["y_treino"].value_counts().index[0]
    previstoBase = np.array([classeMaisFrequente] * len(dados["y_teste"]))
    metricasBase, _ = avaliarClassificador(f"Baseline (sempre '{classeMaisFrequente}')", dados["y_teste"], previstoBase)

    rf = treinarRandomForest(dados, pastaSaida)
    knn = treinarKnn(dados, pastaSaida)

    dfMetricas = pd.DataFrame([metricasBase, rf["metricas"], knn["metricas"]])
    dfMetricas.to_csv(os.path.join(pastaSaida, "modelos_metricas_teste.csv"), index=False)
    print("\n--- Comparação dos classificadores no conjunto de teste ---")
    print(dfMetricas.to_string(index=False))

    kmeans = agruparKmeans(df, pastaSaida)

    return {"random_forest": rf, "knn": knn, "metricas": dfMetricas, "kmeans": kmeans}
