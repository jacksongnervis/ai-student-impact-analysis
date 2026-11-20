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

VARIAVEIS_NOMINAIS = ["Major_Category", "Primary_Use_Case", "Institutional_Policy"]

PROFUNDIDADES_RF = [5, 10, 15, 20, None]
VALORES_K_KNN = [1, 5, 15, 25, 51, 101, 151, 201]

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

        categorias = sorted(df[col].unique())
        for codigo, categoria in enumerate(categorias):
            X.loc[df[col] == categoria, col + "_Cod"] = codigo

    return X


def dividirDados(X, y):
    """Treino 70% / teste 30%; o treino é subdividido em ajuste 80% / validação 20%."""
    XTreino, XTeste, yTreino, yTeste = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=SEMENTE,
        stratify=y,
    )

    XAjuste, XValid, yAjuste, yValid = train_test_split(
        XTreino,
        yTreino,
        test_size=0.2,
        random_state=SEMENTE,
        stratify=yTreino,
    )

    print(f"\nTreino: {len(XTreino)} (ajuste {len(XAjuste)} + validação {len(XValid)}) | Teste: {len(XTeste)}")

    return {
        "X_treino": XTreino,
        "y_treino": yTreino,
        "X_teste": XTeste,
        "y_teste": yTeste,
        "X_ajuste": XAjuste,
        "y_ajuste": yAjuste,
        "X_valid": XValid,
        "y_valid": yValid,
    }


def padronizar(XReferencia, X):
    """Escore Z: (x - média) / desvio-padrão, com média e desvio da base de referência."""
    return (X - XReferencia.mean()) / XReferencia.std()


def avaliarClassificador(nome, yReal, yPrevisto):
    """Matriz de confusão e métricas por classe (precisão, recall, F1) e médias macro."""
    real = yReal.to_numpy()

    linhas = []
    for classeReal in CLASSES_BURNOUT:
        linha = []
        for classePrev in CLASSES_BURNOUT:
            quantidade = ((real == classeReal) & (yPrevisto == classePrev)).sum()
            linha.append(int(quantidade))
        linhas.append(linha)

    nomesLinhas = []
    nomesColunas = []
    for c in CLASSES_BURNOUT:
        nomesLinhas.append(f"Real_{c}")
        nomesColunas.append(f"Previsto_{c}")

    matriz = pd.DataFrame(linhas, index=nomesLinhas, columns=nomesColunas)

    precisoes = []
    recalls = []
    f1s = []
    for i in range(len(CLASSES_BURNOUT)):
        tp = matriz.iloc[i, i]
        fp = matriz.iloc[:, i].sum() - tp
        fn = matriz.iloc[i, :].sum() - tp

        if (tp + fp) > 0:
            precisao = tp / (tp + fp)
        else:
            precisao = 0.0

        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        if (precisao + recall) > 0:
            f1 = 2 * precisao * recall / (precisao + recall)
        else:
            f1 = 0.0

        precisoes.append(precisao)
        recalls.append(recall)
        f1s.append(f1)

    acertos = 0
    for i in range(len(CLASSES_BURNOUT)):
        acertos = acertos + matriz.iloc[i, i]

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
    melhorProfundidade = None
    melhorAcuracia = -1.0

    for profundidade in PROFUNDIDADES_RF:
        modelo = RandomForestClassifier(n_estimators=100, max_depth=profundidade, random_state=SEMENTE)
        modelo.fit(dados["X_ajuste"], dados["y_ajuste"])

        accAjuste = accuracy_score(dados["y_ajuste"], modelo.predict(dados["X_ajuste"]))
        accValid = accuracy_score(dados["y_valid"], modelo.predict(dados["X_valid"]))

        validacao.append({
            "max_depth": str(profundidade),
            "Acuracia_Ajuste": round(accAjuste, 4),
            "Acuracia_Validacao": round(accValid, 4),
        })

        if accValid > melhorAcuracia:
            melhorProfundidade = profundidade
            melhorAcuracia = accValid

    dfValidacao = pd.DataFrame(validacao)
    dfValidacao.to_csv(os.path.join(pastaSaida, "rf_validacao_profundidade.csv"), index=False)

    print("\n--- Random Forest: escolha de max_depth ---")
    print(dfValidacao.to_string(index=False))
    print(f"max_depth escolhido: {melhorProfundidade}")

    modelo = RandomForestClassifier(n_estimators=100, max_depth=melhorProfundidade, random_state=SEMENTE)
    modelo.fit(dados["X_treino"], dados["y_treino"])
    previsto = modelo.predict(dados["X_teste"])

    nomeModelo = f"Random Forest (max_depth={melhorProfundidade})"
    metricas, matriz = avaliarClassificador(nomeModelo, dados["y_teste"], previsto)
    matriz.to_csv(os.path.join(pastaSaida, "rf_matriz_confusao.csv"))

    importancias = pd.DataFrame({
        "Variavel": dados["X_treino"].columns,
        "Importancia": modelo.feature_importances_.round(4),
    })
    importancias = importancias.sort_values(by="Importancia", ascending=False)
    importancias.to_csv(os.path.join(pastaSaida, "rf_importancia_variaveis.csv"), index=False)

    print("--- Random Forest: importância das variáveis ---")
    print(importancias.to_string(index=False))

    return {
        "metricas": metricas,
        "matriz": matriz,
        "validacao": dfValidacao,
        "importancias": importancias,
    }


def treinarKnn(dados, pastaSaida):
    """Escolhe K na validação (atributos padronizados), treina no treino completo e avalia no teste."""
    XAjusteOriginal = dados["X_ajuste"][VARIAVEIS_NUMERICAS]
    XValidOriginal = dados["X_valid"][VARIAVEIS_NUMERICAS]

    XAjuste = padronizar(XAjusteOriginal, XAjusteOriginal)
    XValid = padronizar(XAjusteOriginal, XValidOriginal)

    validacao = []
    melhorK = None
    melhorAcuracia = -1.0

    for k in VALORES_K_KNN:
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(XAjuste, dados["y_ajuste"])

        accValid = accuracy_score(dados["y_valid"], modelo.predict(XValid))
        validacao.append({"K": k, "Acuracia_Validacao": round(accValid, 4)})

        if accValid > melhorAcuracia:
            melhorK = k
            melhorAcuracia = accValid

    dfValidacao = pd.DataFrame(validacao)
    dfValidacao.to_csv(os.path.join(pastaSaida, "knn_validacao_k.csv"), index=False)

    print("\n--- KNN: escolha de K ---")
    print(dfValidacao.to_string(index=False))
    print(f"K escolhido: {melhorK}")

    XTreinoOriginal = dados["X_treino"][VARIAVEIS_NUMERICAS]
    XTesteOriginal = dados["X_teste"][VARIAVEIS_NUMERICAS]

    XTreino = padronizar(XTreinoOriginal, XTreinoOriginal)
    XTeste = padronizar(XTreinoOriginal, XTesteOriginal)

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
    horasPorGrupo = dfGrupos.groupby("Grupo_Original")["Weekly_GenAI_Hours"].mean()
    ordem = horasPorGrupo.sort_values().index

    dfGrupos["Grupo"] = 0
    for novo, original in enumerate(ordem, start=1):
        dfGrupos.loc[dfGrupos["Grupo_Original"] == original, "Grupo"] = novo

    numeros = list(range(1, K_GRUPOS + 1))
    colunasPerfil = VARIAVEIS_KMEANS + ["Skill_Retention_Score", "GPA_Variation"]

    perfil = dfGrupos.groupby("Grupo")[colunasPerfil].mean()
    perfil = perfil.loc[numeros].round(2)
    perfil.insert(0, "n", dfGrupos["Grupo"].value_counts().loc[numeros])

    for nivel in CLASSES_BURNOUT:
        temNivel = dfGrupos["Burnout_Risk_Level"] == nivel
        proporcao = temNivel.groupby(dfGrupos["Grupo"]).mean().loc[numeros]
        perfil[f"Pct_Burnout_{nivel}"] = (proporcao * 100).round(1)

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
    nomeBase = f"Baseline (sempre '{classeMaisFrequente}')"
    metricasBase, _ = avaliarClassificador(nomeBase, dados["y_teste"], previstoBase)

    rf = treinarRandomForest(dados, pastaSaida)
    knn = treinarKnn(dados, pastaSaida)

    dfMetricas = pd.DataFrame([metricasBase, rf["metricas"], knn["metricas"]])
    dfMetricas.to_csv(os.path.join(pastaSaida, "modelos_metricas_teste.csv"), index=False)

    print("\n--- Comparação dos classificadores no conjunto de teste ---")
    print(dfMetricas.to_string(index=False))

    kmeans = agruparKmeans(df, pastaSaida)

    return {"random_forest": rf, "knn": knn, "metricas": dfMetricas, "kmeans": kmeans}
