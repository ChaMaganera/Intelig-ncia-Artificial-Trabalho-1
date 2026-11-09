# Intelig-ncia-Artificial-Trabalho-1

Trabalho 1 de Inteligência Artificial — Q3.2026.

O projeto compara os modelos Naive Bayes e MLP nos conjuntos Cats and Dogs e Dry Bean. Os dados devem estar nas pastas `cat_dog_files` e `dry_bean_files`, na raiz do projeto.

## Requisitos

- Python 3.10 ou superior
- pip

## Como instalar e executar

Abra um terminal na pasta do projeto e crie um ambiente virtual:

```powershell
py -m venv .venv
```

Ative o ambiente virtual no Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

No macOS ou Linux, use:

```bash
source .venv/bin/activate
```

Instale as bibliotecas necessárias:

```bash
python -m pip install numpy pandas matplotlib scikit-learn
```

Execute os quatro experimentos e gere os gráficos:

```bash
python main.py
```

O programa imprime as métricas e abre gráficos com as matrizes de confusão e a comparação dos modelos. Feche cada janela de gráfico para continuar a execução.

## Arquivos de dados
Os CSVs e as imagens já estão incluídos no repositório. Caso precise baixar novamente os conjuntos ou analisar a base de dados:

- [Cats and Dogs](https://www.kaggle.com/datasets/marquis03/cats-and-dogs)
- [Dry Bean](https://www.kaggle.com/datasets/sansuthi/dry-bean-dataset)

Documentação do trabalho: [Google Docs](https://docs.google.com/document/d/1rNVYjYJMaRFjRdJnOthwwoaCayldoqLoWmOGw2WYzu4/edit?usp=sharing).
