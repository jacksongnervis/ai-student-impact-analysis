"""Orquestrador da pipeline de Ciência de Dados (Milestone 1)."""

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


def log(etapa: str, mensagem: str) -> None:
    """Emite log no terminal com timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{etapa}] {mensagem}")


def main() -> None:
    """Executa a pipeline de análise de dados de ponta a ponta."""
    log("ETAPA 1/4", "Carregando base de dados...")
    df_raw = carregar_dados()
    log("ETAPA 1/4", f"Sucesso: {len(df_raw)} registros carregados.")

    log("ETAPA 2/4", "Auditando e preparando dados...")
    auditar_integridade(df_raw)
    df = preparar_dados(df_raw)
    log("ETAPA 2/4", "Sucesso: auditoria concluída e tipos preparados.")

    log("ETAPA 3/4", "Calculando métricas descritivas e frequências...")
    calcular_metricas_quantitativas(df)
    calcular_frequencias_qualitativas(df)
    log("ETAPA 3/4", "Sucesso: relatórios CSV exportados.")

    log("ETAPA 4/4", "Gerando figuras estatísticas a 300 DPI...")
    gerar_graficos_distribuicao(df)
    log("ETAPA 4/4", "Sucesso: 9 figuras geradas em 'figures/'.")

    print("\nPipeline concluída com sucesso.")


if __name__ == "__main__":
    main()
