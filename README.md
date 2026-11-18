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
│   ├── bivariate_analysis.py       # Correlação, regressão linear simples, médias por grupo e qui-quadrado (Milestone 2)
│   ├── models.py                   # Random Forest, KNN e K-means (Milestone 2)
│   └── results_visualization.py    # Gráficos dos resultados (Milestone 2)
├── main.py                         # Executa a pipeline completa (7 etapas)
├── analise_completa.ipynb          # Notebook que executa a pipeline e exibe tabelas e gráficos
├── requirements.txt                # Dependências do projeto
├── .gitignore
└── README.md
```

## 2. Como executar

### 2.1 Google Colab

1. Abra o `analise_completa.ipynb` no Colab (`Arquivo > Fazer upload de notebook` ou `Arquivo > Abrir notebook > GitHub`).
2. Clique em `Ambiente de execução > Executar tudo`.

A primeira célula baixa o repositório e instala as dependências. Os gráficos aparecem abaixo de cada célula.

### 2.2 VS Code (Linux)

Pré-requisito no Ubuntu/Debian (uma única vez):
```bash
sudo apt install python3-venv python3-pip
```

Na pasta do projeto:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Opção A – script:** `python main.py` gera todas as tabelas em `data/processed/` e todos os gráficos em `figures/` (abra os PNGs pelo explorador do VS Code).

**Opção B – notebook:** abra o `analise_completa.ipynb`, selecione o interpretador `venv` no canto superior direito e clique em *Run All*. Na primeira vez o VS Code pedirá para instalar a extensão *Jupyter* e o pacote `ipykernel`; aceite.

### 2.3 Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## 3. Etapas da pipeline (`main.py`)

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

## 4. Resultados gerados

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
