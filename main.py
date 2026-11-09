import os
from cat_dog_NaiveBayes import cat_dog_NaiveBayes
from cat_dog_MLP import cat_dog_MLP
from dry_bean_NaiveBayes import dry_bean_NaiveBayes
from dry_bean_MLP import dry_bean_MLP
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
      
diretorio_projeto = os.path.dirname(os.path.abspath(__file__))

cat_dog_diretorio_raiz = os.path.join(diretorio_projeto, "cat_dog_files")
cat_dog_treino = os.path.join(cat_dog_diretorio_raiz, "train.csv")
cat_dog_teste = os.path.join(cat_dog_diretorio_raiz, "val.csv")

pasta_dry_bean = os.path.join(diretorio_projeto, "dry_bean_files")
csv_dry_bean = os.path.join(pasta_dry_bean, "Dry_Bean.csv")

df_modelos = [
    [cat_dog_diretorio_raiz, cat_dog_treino, cat_dog_teste],
    [cat_dog_diretorio_raiz, cat_dog_treino, cat_dog_teste],
    [csv_dry_bean],
    [csv_dry_bean]
]

#Criando uma nested class para organizar as métricas
@dataclass
class Metricas:
    acuracia: float 
    precisao: float
    recall: float
    f1: float


def gerar_grafico(resultados):

    modelos = [
        "NB Cat/Dog",
        "MLP Cat/Dog",
        "NB Dry Bean",
        "MLP Dry Bean"
    ]

    metricas = ["Acurácia", "Precisão", "Recall", "F1-Score"]

    valores = np.array([
        [r.acuracia, r.precisao, r.recall, r.f1]
        for r in resultados
    ])

    x = np.arange(len(metricas))
    largura = 0.2

    plt.figure(figsize=(10, 6))

    for i in range(len(modelos)):
        plt.bar(
            x + i * largura,
            valores[i],
            largura,
            label=modelos[i]
        )

    plt.xlabel("Métricas")
    plt.ylabel("Valor")
    plt.title("Comparação das métricas dos modelos")
    plt.xticks(x + largura * 1.5, metricas)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
def main():
    # Chamando os modelos
   metricas_nb_cat = cat_dog_NaiveBayes(*df_modelos[0])
   metricas_mlp_cat = cat_dog_MLP(*df_modelos[1])

   metricas_nb_bean = dry_bean_NaiveBayes(*df_modelos[2])
   metricas_mlp_bean = dry_bean_MLP(*df_modelos[3])

   # Armazenando as métricas em objetos
   resultados = [
       Metricas(*metricas_nb_cat),
       Metricas(*metricas_mlp_cat),
       Metricas(*metricas_nb_bean),
       Metricas(*metricas_mlp_bean)
   ]
   return resultados
    
if __name__ == "__main__":
    gerar_grafico(resultados = main())
