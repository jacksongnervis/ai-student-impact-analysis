"""Auditoria de integridade, conversão de tipos e engenharia de atributos."""

from __future__ import annotations

from typing import Any, Dict
import pandas as pd


def auditar_integridade(df: pd.DataFrame) -> Dict[str, Any]:
    """Valida ausência de nulos, duplicatas, unicidade de ID e restrições de domínio."""
    nulos = df.isnull().sum()
    duplicados = int(df.duplicated().sum())
    unicos_id = int(df["Student_ID"].nunique())

    dominio = {
        "Pre_Semester_GPA [0.0, 4.0]": bool(df["Pre_Semester_GPA"].between(0.0, 4.0).all()),
        "Post_Semester_GPA [0.0, 4.0]": bool(df["Post_Semester_GPA"].between(0.0, 4.0).all()),
        "Anxiety_Level [1, 10]": bool(df["Anxiety_Level_During_Exams"].between(1, 10).all()),
        "AI_Dependency [1, 10]": bool(df["Perceived_AI_Dependency"].between(1, 10).all()),
        "Weekly_GenAI_Hours >= 0": bool((df["Weekly_GenAI_Hours"] >= 0).all()),
        "Traditional_Study_Hours >= 0": bool((df["Traditional_Study_Hours"] >= 0).all()),
    }

    print("\n--- Auditoria de Integridade ---")
    print(f"Total de registros: {len(df):,} | Linhas duplicadas: {duplicados}".replace(",", "."))
    print(f"IDs únicos: {unicos_id:,} (Duplicados: {len(df) - unicos_id})".replace(",", "."))
    print("Valores nulos por coluna:")
    nulos_existentes = nulos[nulos > 0]
    print(nulos_existentes if not nulos_existentes.empty else "  Nenhum valor nulo identificado.")
    print("Validação de domínio:")
    for regra, valido in dominio.items():
        print(f"  [{'OK' if valido else 'FALHA'}] {regra}")

    return {
        "nulos": nulos.to_dict(),
        "duplicados": duplicados,
        "unicos_id": unicos_id,
        "validacao_dominio": dominio,
    }


def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica tipagem categórica (ordinal e nominal) e deriva GPA_Variation."""
    df_proc = df.copy()

    df_proc["Year_of_Study"] = pd.Categorical(
        df_proc["Year_of_Study"],
        categories=["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
        ordered=True,
    )
    df_proc["Prompt_Engineering_Skill"] = pd.Categorical(
        df_proc["Prompt_Engineering_Skill"],
        categories=["Beginner", "Intermediate", "Advanced"],
        ordered=True,
    )
    df_proc["Burnout_Risk_Level"] = pd.Categorical(
        df_proc["Burnout_Risk_Level"],
        categories=["Low", "Medium", "High"],
        ordered=True,
    )

    for col in ["Major_Category", "Primary_Use_Case", "Institutional_Policy"]:
        df_proc[col] = df_proc[col].astype("category")

    df_proc["Paid_Subscription"] = df_proc["Paid_Subscription"].astype(bool)
    df_proc["GPA_Variation"] = df_proc["Post_Semester_GPA"] - df_proc["Pre_Semester_GPA"]

    return df_proc
