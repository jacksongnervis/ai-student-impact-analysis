"""Orquestrador da pipeline de Ciência de Dados (Milestones 1 e 2)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Adiciona o diretório raiz ao sys.path para importações locais
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import carregar_dados
from src.audit_cleaning import auditar_integridade, preparar_dados
from src.descriptive_metrics import (
    calcular_metricas_quantitativas,
    calcular_frequencias_qualitativas,
)
from src.visualization import gerar_graficos_distribuicao
from src.bivariate_analysis import executar_analise_bivariada
from src.models import executar_modelos
from src.results_visualization import gerar_graficos_resultados


def log(etapa: str, mensagem: str) -> None:
    """Emite log no terminal com timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{etapa}] {mensagem}")


def main() -> None:
    """Executa a pipeline de análise de dados de ponta a ponta."""
    log("ETAPA 1/7", "Carregando base de dados...")
    df_raw = carregar_dados()
    log("ETAPA 1/7", f"Sucesso: {len(df_raw)} registros carregados.")

    log("ETAPA 2/7", "Auditando e preparando dados...")
    auditar_integridade(df_raw)
    df = preparar_dados(df_raw)
    log("ETAPA 2/7", "Sucesso: auditoria concluída e atributos derivados criados.")

    log("ETAPA 3/7", "Calculando métricas descritivas e frequências...")
    calcular_metricas_quantitativas(df)
    calcular_frequencias_qualitativas(df)
    log("ETAPA 3/7", "Sucesso: relatórios CSV exportados.")

    log("ETAPA 4/7", "Gerando figuras de distribuição a 300 DPI...")
    gerar_graficos_distribuicao(df)
    log("ETAPA 4/7", "Sucesso: 11 figuras de distribuição geradas em 'figures/'.")

    log("ETAPA 5/7", "Análise de duas variáveis (correlação, regressão, médias e qui-quadrado)...")
    bivariada = executar_analise_bivariada(df)
    log("ETAPA 5/7", "Sucesso: tabelas exportadas em 'data/processed/'.")

    log("ETAPA 6/7", "Treinando e avaliando modelos (Random Forest, KNN e K-means)...")
    modelos = executar_modelos(df)
    log("ETAPA 6/7", "Sucesso: métricas dos modelos exportadas em 'data/processed/'.")

    log("ETAPA 7/7", "Gerando figuras dos resultados a 300 DPI...")
    gerar_graficos_resultados(df, bivariada, modelos)
    log("ETAPA 7/7", "Sucesso: 17 figuras de resultados geradas em 'figures/'.")

    print("\nPipeline concluída com sucesso.")


if __name__ == "__main__":
    main()
