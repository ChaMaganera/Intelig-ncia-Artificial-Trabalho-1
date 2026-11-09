import os
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
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
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        elif len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
                
        # Redimensionamento da imagem
        linhas_originais, colunas_originais = img.shape[:2]

        indices_linhas = np.linspace(0, linhas_originais - 1, 64).astype(int)
        indices_colunas = np.linspace(0, colunas_originais - 1, 64).astype(int)

        img_redimensionada = img[np.ix_(indices_linhas, indices_colunas)]

        X_list.append(img_redimensionada.flatten())
        y_list.append(label)

    return np.array(X_list), np.array(y_list)


def cat_dog_NaiveBayes(pasta_base, csv_treino, csv_teste):
    # Lendo os dados de treino
    X_treino, y_treino = carregar_e_processar_dados(csv_treino, pasta_base)
    
    # Lendo os dados de teste
    X_teste, y_teste = carregar_e_processar_dados(csv_teste, pasta_base)
    
    # Normalizacao dos dados
    scaler = StandardScaler()
    X_treino_norm = scaler.fit_transform(X_treino)
    X_teste_norm = scaler.transform(X_teste)
    
    #Hiper parâmetro var_smoothing
    nb_modelo = GaussianNB(var_smoothing=10e-1)
    nb_modelo.fit(X_treino_norm, y_treino)
    
    # Testando o modelo
    y_pred_nb = nb_modelo.predict(X_teste_norm)

    # Calculando métricas
    acuracia = accuracy_score(y_teste, y_pred_nb) 
    precisao = precision_score(y_teste, y_pred_nb, pos_label=nb_modelo.classes_[1])
    recall = recall_score(y_teste,  y_pred_nb, pos_label=nb_modelo.classes_[1])
    f1 = f1_score(y_teste, y_pred_nb, pos_label=nb_modelo.classes_[1])
    
    print("================================================")
    print("DATASET: CATS AND DOGS | MODELO: NAIVE BAYES")
    print("================================================")
    
    print("MÉTRICAS:")
    print("Acurácia Naive Bayes cat_dog:", acuracia)
    print("Precisão Naive Bayes no cat_dog:", precisao)
    print("Recall Naive Bayes no cat_dog:", recall)
    print("F1-Score Naive Bayes no cat_dog:", f1)
    
    # Relatorio
    print("\nRelatório Naive Bayes cat_dog:\n", classification_report(y_teste, y_pred_nb))
    
    # Matriz de Confusão
    matriz = confusion_matrix(y_teste, y_pred_nb)
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
    disp.plot(cmap='Oranges')
    
    plt.title("Matriz de Confusão - Naive Bayes (Cats & Dogs)")
    plt.show()
    print("\n\n\n\n")
    
    return acuracia, precisao, recall, f1