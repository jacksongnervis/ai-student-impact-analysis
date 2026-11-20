# Análise do Impacto do uso de IA em estudantes

Projeto de Ciência de Dados que analisa os impactos do uso de Inteligência Artificial Generativa no desempenho acadêmico, nos hábitos de estudo, na retenção de habilidades, na ansiedade, na dependência percebida e no risco de burnout de estudantes de ensino superior.

Base de dados: *AI Impact on Students* (Kaggle) — 50.000 registros e 16 colunas.

## 1. Estrutura do Projeto

```text
ai-student-impact-analysis/
├── data/
│   ├── raw/                        # Base bruta (baixada automaticamente via gdown)
│   └── processed/                  # Tabelas CSV geradas pela pipeline
├── figures/                        # Gráficos PNG gerados pela pipeline (300 DPI)
├── src/
│   ├── __init__.py                 # Inicializador do pacote Python
│   ├── data_loader.py              # Download via gdown e leitura da base bruta
│   ├── audit_cleaning.py           # Auditoria, codificação ordinal, faixas de horas de IA e GPA_Variation
│   ├── descriptive_metrics.py      # Métricas descritivas e tabelas de frequência (Milestone 1)
│   ├── visualization.py            # Pizzas, histogramas e curvas KDE (Milestone 1)
│   ├── bivariateAnalysis.py        # Correlação, regressão linear simples, médias por grupo e qui-quadrado (Milestone 2)
│   ├── models.py                   # Random Forest, KNN e K-means (Milestone 2)
│   └── resultsVisualization.py     # Gráficos dos resultados (Milestone 2)
├── main.py                         # Executa a pipeline completa (7 etapas)
├── requirements.txt                # Dependências do projeto
├── .gitignore
└── README.md
```

## 2. Instalação e Configuração

### 2.1 Criar e ativar o ambiente virtual 

No Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

No Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2.2 Instalar dependências

A partir do diretório raiz do projeto:
```bash
pip install -r requirements.txt
```

## 3. Execução da Pipeline

A execução de toda a esteira de ponta a ponta é orquestrada pelo script `main.py`:

```bash
python main.py
```

## 4. Etapas da pipeline (`main.py`)

| Etapa | Descrição |
|---|---|
| 1 | Carregamento da base (download automático se não existir) |
| 2 | Auditoria (nulos, duplicatas, domínios) e preparação (codificação ordinal, faixas de horas de IA, GPA_Variation) |
| 3 | Métricas descritivas e frequências |
| 4 | Gráficos de distribuição |
| 5 | Análise de duas variáveis: correlação de Pearson, regressão linear simples, médias por grupo, tabelas de contingência, qui-quadrado e coeficiente de contingência |
| 6 | Classificação do risco de burnout (baseline, Random Forest e KNN; divisão 70/30 estratificada com validação para escolha de hiperparâmetros) e agrupamento K-means (método do cotovelo) |
| 7 | Gráficos dos resultados |

Os resultados são reprodutíveis: todos os algoritmos usam semente fixa (`random_state=42`).

## 5. Resultados gerados

### Tabelas (`data/processed/`)

| Arquivo | Conteúdo |
|---|---|
| `metricas_descritivas_numericas.csv` | Média, mediana, desvio-padrão, mínimo, quartis e máximo |
| `frequencias_variaveis_categoricas.csv` | Frequências absolutas e relativas |
| `correlacoes_pearson.csv` | Matriz de correlação de Pearson |
| `regressoes_lineares_simples.csv` | b0, b1, r e R² de cada regressão |
| `medias_por_<variável>.csv` | Médias das variáveis principais por grupo (8 arquivos) |
| `contingencia_Burnout_Risk_Level_por_<variável>.csv` | % de cada nível de burnout por categoria (7 arquivos) |
| `qui_quadrado_resumo.csv` | Qui-quadrado e coeficiente de contingência C de cada variável com o burnout |
| `rf_validacao_profundidade.csv` / `knn_validacao_k.csv` | Escolha dos hiperparâmetros na validação |
| `rf_matriz_confusao.csv` / `knn_matriz_confusao.csv` | Matrizes de confusão no teste |
| `rf_importancia_variaveis.csv` | Importância das variáveis no Random Forest |
| `modelos_metricas_teste.csv` | Acurácia, precisão, recall e F1 (macro e por classe) no teste |
| `kmeans_cotovelo.csv` / `kmeans_perfil_grupos.csv` | Inércia por k e perfil dos grupos |

### Gráficos (`figures/`)

**Milestone 1 – distribuições:** `distribuicao_major_category.png`, `distribuicao_burnout_risk.png` e `vertical_*.png` (histograma + KDE de 9 variáveis).

**Milestone 2 – resultados:**

| Arquivo | Conteúdo |
|---|---|
| `m2_h1_dispersao_horas_variacao_gpa.png` | Horas de IA × variação do GPA, com reta de regressão |
| `m2_h1_boxplot_variacao_por_faixa.png` | Variação do GPA por faixa de horas de IA |
| `m2_h1_dispersao_pre_pos_por_horas.png` | GPA inicial × final colorido pelas horas de IA |
| `m2_h1_boxplot_variacao_por_uso_e_prompt.png` | Variação do GPA por finalidade de uso e habilidade em prompts |
| `m2_correlacao_heatmap.png` | Mapa de calor da matriz de correlação |
| `m2_h2_dispersoes_regressao.png` | Quatro dispersões com reta de regressão |
| `m2_h2_medias_por_faixa.png` | Médias de estudo, retenção, ansiedade e dependência por faixa de horas |
| `m2_h2_boxplot_por_burnout.png` | Horas, dependência e ansiedade por nível de burnout |
| `m2_h2_burnout_por_faixa.png` | % de cada nível de burnout por faixa de horas |
| `m2_h2_associacao_burnout.png` | Coeficiente de contingência de cada variável com o burnout |
| `m2_rf_overfitting_profundidade.png` | Acurácia de treino × validação por profundidade (overfitting) |
| `m2_knn_validacao_k.png` | Acurácia de validação por K |
| `m2_modelos_comparacao.png` | Métricas de baseline, Random Forest e KNN no teste |
| `m2_rf_importancia_variaveis.png` | Importância das variáveis no Random Forest |
| `m2_kmeans_cotovelo.png` | Método do cotovelo |
| `m2_kmeans_grupos.png` | Características dos grupos do K-means |
| `m2_kmeans_burnout_por_grupo.png` | % de cada nível de burnout por grupo |
