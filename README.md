# Análise do Impacto uso de IA em estudantes

Projeto estruturado de Ciência de Dados voltado à análise de impactos do uso de Inteligência Artificial Generativa no desempenho acadêmico, ansiedade, dependência percebida e risco de burnout em estudantes de ensino superior.

## 1. Estrutura do Projeto

O projeto adota uma arquitetura modular que separa a ingestão de dados, sanitização, cálculo estatístico e visualização de dados:

```text
projeto-ciencia-dados/
├── data/
│   ├── raw/                      # Destino do download automático da base
│   └── processed/                # Destino dos CSVs com métricas geradas
├── figures/                      # Destino das imagens de gráficos geradas
├── src/
│   ├── __init__.py               # Inicializador do pacote Python
│   ├── data_loader.py            # Download via gdown e leitura da base bruta
│   ├── audit_cleaning.py         # Auditoria de integridade, parsers de tipos, etc
│   ├── descriptive_metrics.py    # Cálculo e exportação de métricas
│   └── visualization.py          # Geração dos gráficos de pizza, histogramas e KDE
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



