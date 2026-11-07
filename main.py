import os
from cat_dog_NaiveBayes import cat_dog_NaiveBayes
from cat_dog_MLP import cat_dog_MLP
from dry_bean_NaiveBayes import dry_bean_NaiveBayes
from dry_bean_MLP import dry_bean_MLP
from dataclasses import dataclass
    
cat_dog_diretorio_raiz = r"./cat_dog_files"
cat_dog_treino = os.path.join(cat_dog_diretorio_raiz, "train.csv")
cat_dog_teste = os.path.join(cat_dog_diretorio_raiz, "val.csv")

pasta_dry_bean = r"./dry_bean_files"
csv_dry_bean = os.path.join(pasta_dry_bean, "dry_bean.csv")

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
    resultados = main()

