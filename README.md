# Análise do Impacto do uso de IA em estudantes

Projeto estruturado de Ciência de Dados voltado à análise de impactos do uso de Inteligência Artificial Generativa no desempenho acadêmico, ansiedade, dependência percebida e risco de burnout em estudantes de ensino superior.

## 1. Estrutura do Projeto

O projeto adota uma arquitetura modular que separa a ingestão de dados, sanitização, cálculo estatístico e visualização de dados:

```text
ai-student-impact-analysis/
├── data/
│   ├── raw/                      # Destino do download automático da base
│   └── processed/                # Destino dos CSVs com métricas geradas
├── figures/                      # Destino das imagens de gráficos geradas
├── src/
│   ├── __init__.py               # Inicializador do pacote Python
│   ├── data_loader.py            # Download via gdown e leitura da base bruta
│   ├── audit_cleaning.py         # Auditoria de integridade, parsers de tipos, etc
│   ├── descriptive_metrics.py    # Cálculo e exportação de métricas
│   ├── visualization.py          # Geração dos gráficos de pizza, histogramas e KDE
│   ├── bivariateAnalysis.py      # Correlação, regressão linear, médias por grupo e qui-quadrado
│   ├── models.py                 # Treinamento e avaliação do Random Forest, KNN e K-means
│   └── resultsVisualization.py   # Geração dos gráficos de resultados das hipóteses e modelos
├── main.py                       # Orquestrador da pipeline completa de execução
├── requirements.txt              # Dependências do projeto com especificações de versão
├── .gitignore                    # Regras de exclusão para Python e ambientes virtuais
└── README.md                     # Documentação de reprodução e estrutura
```

---

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

---

## 3. Execução da Pipeline

A execução de toda a esteira de ponta a ponta é orquestrada pelo script `main.py`:

```bash
python main.py
```

---

## 4. Resultados Gerados

### Tabelas (`data/processed/`)
- `metricas_descritivas_numericas.csv`: Métricas descritivas univariadas.
- `frequencias_variaveis_categoricas.csv`: Frequências absolutas e percentuais.
- `correlacoes_pearson.csv`: Matriz de correlação de Pearson entre as variáveis quantitativas.
- `regressoes_lineares_simples.csv`: Coeficientes b0 e b1, r e R² de cada regressão linear simples.
- `medias_por_<variavel>.csv`: Médias das variáveis principais em cada grupo da variável.
- `contingencia_Burnout_Risk_Level_por_<variavel>.csv`: Percentual de cada nível de burnout em cada categoria da variável.
- `qui_quadrado_resumo.csv`: Qui-quadrado e coeficiente de contingência C de cada variável com o burnout.
- `rf_validacao_profundidade.csv`: Acurácia de treino e validação do Random Forest para cada profundidade.
- `rf_matriz_confusao.csv`: Matriz de confusão do Random Forest no teste.
- `rf_importancia_variaveis.csv`: Importância de cada variável no Random Forest.
- `knn_validacao_k.csv`: Acurácia de validação do KNN para cada valor de K.
- `knn_matriz_confusao.csv`: Matriz de confusão do KNN no teste.
- `modelos_metricas_teste.csv`: Acurácia, precisão, recall e F1 do baseline, Random Forest e KNN no teste.
- `kmeans_cotovelo.csv`: Inércia do K-means para cada número de grupos.
- `kmeans_perfil_grupos.csv`: Perfil médio e percentual de burnout de cada grupo do K-means.

### Gráficos (`figures/`)
- `distribuicao_major_category.png`: Proporção por área acadêmica.
- `distribuicao_burnout_risk.png`: Proporção do risco de burnout.
- `vertical_pre_gpa.png`: Histograma + KDE do GPA pré-semestre.
- `vertical_post_gpa.png`: Histograma + KDE do GPA pós-semestre.
- `vertical_weekly_ai.png`: Histograma + KDE de horas semanais com IA generativa.
- `vertical_anxiety.png`: Histograma + KDE do nível de ansiedade em exames.
- `vertical_traditional_study.png`: Histograma + KDE de horas de estudo tradicional.
- `vertical_ai_dependency.png`: Histograma + KDE do nível de dependência de IA percebida.
- `vertical_skill_retention.png`: Histograma + KDE do score de retenção de habilidades.
- `vertical_gpa_variation.png`: Histograma + KDE da variação do GPA.
- `vertical_tool_diversity.png`: Histograma + KDE da quantidade de ferramentas de IA utilizadas.
- `m2_h1_dispersao_horas_variacao_gpa.png`: Dispersão das horas de IA pela variação do GPA, com reta de regressão.
- `m2_h1_boxplot_variacao_por_faixa.png`: Boxplot da variação do GPA por faixa de horas de IA.
- `m2_h1_dispersao_pre_pos_por_horas.png`: GPA pré-semestre pelo pós-semestre, colorido pelas horas de IA.
- `m2_h1_boxplot_variacao_por_uso_e_prompt.png`: Boxplot da variação do GPA por finalidade de uso e habilidade em prompts.
- `m2_correlacao_heatmap.png`: Mapa de calor da matriz de correlação de Pearson.
- `m2_h2_dispersoes_regressao.png`: Dispersões com reta de regressão entre uso de IA, estudo, retenção e ansiedade.
- `m2_h2_medias_por_faixa.png`: Médias de estudo tradicional, retenção, ansiedade e dependência por faixa de horas.
- `m2_h2_boxplot_por_burnout.png`: Boxplot das horas de IA, dependência e ansiedade por nível de burnout.
- `m2_h2_burnout_por_faixa.png`: Percentual de cada nível de burnout por faixa de horas de IA.
- `m2_h2_associacao_burnout.png`: Coeficiente de contingência de cada variável com o burnout.
- `m2_rf_overfitting_profundidade.png`: Acurácia de treino e validação do Random Forest por profundidade.
- `m2_knn_validacao_k.png`: Acurácia de validação do KNN por valor de K.
- `m2_modelos_comparacao.png`: Comparação das métricas do baseline, Random Forest e KNN no teste.
- `m2_rf_importancia_variaveis.png`: Importância das variáveis no Random Forest.
- `m2_kmeans_cotovelo.png`: Método do cotovelo do K-means.
- `m2_kmeans_grupos.png`: Características dos grupos formados pelo K-means.
- `m2_kmeans_burnout_por_grupo.png`: Percentual de cada nível de burnout por grupo do K-means.



