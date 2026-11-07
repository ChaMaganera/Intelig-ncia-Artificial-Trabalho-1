import os
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,   
    recall_score,      
    f1_score,          
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def carregar_e_processar_dados(caminho_csv, pasta_base):
    """
    Le o CSV e processa as imagens.
    """
    df = pd.read_csv(caminho_csv)
        
    X_list = []
    y_list = []
    
    for index, row in df.iterrows():
        caminho_imagem = os.path.join(pasta_base, row['image:FILE'])
        label = row['category']
        img = mpimg.imread(caminho_imagem)
            
        #Padronizacao dos canais de cor
        #Se tiver 4 canais (RGBA) a matriz é cortada ficando apenas com 3 (RGB)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        #Se tiver 1 canal (tons de cinza) a matriz é empilhada 3 vezes para simular o formato do RGB (matriz 3D) 
        elif len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
                
        # Redimensionamento da imagem
        # Dimensao original da imagem
        linhas_originais, colunas_originais = img.shape[:2]

        #Coordenadas para linhas/colunas uniformemente espaçadas
        indices_linhas = np.linspace(0, linhas_originais - 1, 32).astype(int)
        indices_colunas = np.linspace(0, colunas_originais - 1, 32).astype(int)

        #Cruza as coordenadas da imagem original para extrair apenas os pixels calculados    
        img_redimensionada = img[np.ix_(indices_linhas, indices_colunas)]

        #Achata a matriz tridimensional em um vetor    
        X_list.append(img_redimensionada.flatten())
        y_list.append(label)

    return np.array(X_list), np.array(y_list)

#Caminhos
pasta_base = r".\cat_dog_files"
csv_treino = os.path.join(pasta_base, "train.csv")
csv_teste = os.path.join(pasta_base, "val.csv")

#Lendo os dados de treino
X_treino, y_treino = carregar_e_processar_dados(csv_treino, pasta_base)

#Lendo os dados de teste
X_teste, y_teste = carregar_e_processar_dados(csv_teste, pasta_base)

#Normalizacao dos dados
scaler = StandardScaler()
X_treino_norm = scaler.fit_transform(X_treino)
X_teste_norm = scaler.transform(X_teste)

#Treinando a rede neural
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=300, random_state=42)
mlp.fit(X_treino_norm, y_treino)

#Testando o modelo
y_pred_mlp = mlp.predict(X_teste_norm)

#Calculando métricas
print("\nAcurácia MLP cat_dog:", accuracy_score(y_teste, y_pred_mlp))
print("\nPrecisão MLP cat_dog:", precision_score(y_teste, y_pred_mlp))
print("\nRecall MLP cat_dog:", recall_score(y_teste, y_pred_mlp))
print("\nF1-Score MLP cat_dog:", f1_score(y_teste, y_pred_mlp))

#Relatorio
print("Relatório MLP cat_dog:\n", classification_report(y_teste, y_pred_mlp))

#Matriz de Confusão
matriz = confusion_matrix(y_teste, y_pred_mlp)
disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
disp.plot(cmap='Blues')

plt.title("Matriz de Confusão - MLP (Cats & Dogs)")
plt.show()