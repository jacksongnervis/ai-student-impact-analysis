"""Auditoria de integridade, codificação de categorias e engenharia de atributos."""

from __future__ import annotations

from typing import Any, Dict
import pandas as pd

# Ordem lógica das variáveis categóricas ordinais (da menor para a maior)
ORDEM_CATEGORIAS = {
    "Year_of_Study": ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
    "Prompt_Engineering_Skill": ["Beginner", "Intermediate", "Advanced"],
    "Burnout_Risk_Level": ["Low", "Medium", "High"],
}

# Faixas de horas semanais de uso de IA: (rótulo, início inclusivo, fim exclusivo)
FAIXAS_HORAS_IA = [
    ("0-2 h", 0, 2),
    ("2-5 h", 2, 5),
    ("5-10 h", 5, 10),
    ("10-20 h", 10, 20),
    ("20+ h", 20, 41),
]

ORDEM_FAIXAS = []
for rotulo, inicio, fim in FAIXAS_HORAS_IA:
    ORDEM_FAIXAS.append(rotulo)


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
        "Skill_Retention_Score [0, 100]": bool(df["Skill_Retention_Score"].between(0, 100).all()),
        "Tool_Diversity [1, 5]": bool(df["Tool_Diversity"].between(1, 5).all()),
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
    """Cria codificação ordinal das categorias, faixas de horas de IA e GPA_Variation."""
    df_proc = df.copy()

    # Codificação ordinal: cada categoria recebe sua posição na ordem lógica (0, 1, 2, ...)
    for col, ordem in ORDEM_CATEGORIAS.items():
        df_proc[col + "_Ord"] = 0

        for posicao, categoria in enumerate(ordem):
            df_proc.loc[df_proc[col] == categoria, col + "_Ord"] = posicao

    # Faixas de horas semanais de IA
    df_proc["Faixa_Horas_IA"] = ""
    for rotulo, inicio, fim in FAIXAS_HORAS_IA:
        horas = df_proc["Weekly_GenAI_Hours"]
        filtro = (horas >= inicio) & (horas < fim)
        df_proc.loc[filtro, "Faixa_Horas_IA"] = rotulo

    df_proc["Paid_Subscription"] = df_proc["Paid_Subscription"].astype(bool)
    df_proc["GPA_Variation"] = df_proc["Post_Semester_GPA"] - df_proc["Pre_Semester_GPA"]

    return df_proc
