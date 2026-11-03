"""Módulo de download e carregamento de dados educacionais."""

from __future__ import annotations

import os
import gdown
import pandas as pd


def carregar_dados(
    caminho_arquivo: str = "data/raw/impactosIa.csv",
    file_id: str = "1ot6njhnWqjrCk1VyxnqSeE94ghJKRzmc",
) -> pd.DataFrame:
    """Garante a existência local do dataset e o carrega em um DataFrame."""
    os.makedirs(os.path.dirname(caminho_arquivo) or ".", exist_ok=True)

    if not os.path.exists(caminho_arquivo):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Baixando dataset de: {url}")
        gdown.download(url, caminho_arquivo, quiet=False)

    df = pd.read_csv(caminho_arquivo)
    print(f"Base carregada: {len(df):,} registros e {df.shape[1]} colunas.".replace(",", "."))
    return df
