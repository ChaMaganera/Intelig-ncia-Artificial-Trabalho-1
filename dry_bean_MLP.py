import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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


def dry_bean_MLP(csv_file):
    # Lendo a base
    df = pd.read_csv(csv_file)
    
    # Separando os dados de entrada (features) e o gabarito (target)
    features = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "AspectRation",
        "Eccentricity",
        "ConvexArea",
        "EquivDiameter",
        "Extent",
        "Solidity",
        "roundness",
        "Compactness",
        "ShapeFactor1",
        "ShapeFactor2",
        "ShapeFactor3",
        "ShapeFactor4",
    ]
    x = df[features]
    y = df["Class"]
    
    #Separando os dados de teste dos dados de treino
    #teste_size define quantos % da base sera dedica a testes(no caso 20% para testes e 80% para treino)
    #random_state define a seed para a aleatorização dos testes
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.2, random_state = 42)
    
    # Normalizando os dados
    scaler = StandardScaler()
    x_treino_norm = scaler.fit_transform(x_treino)
    x_teste_norm = scaler.transform(x_teste)
    
    # Treinando a rede neural
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=300, random_state=42)
    mlp.fit(x_treino_norm, y_treino)
    
    # Testando o modelo
    y_pred_mlp = mlp.predict(x_teste_norm)
    
    # Calculando métricas
    acuracia = accuracy_score(y_teste,y_pred_mlp) 
    precisao = precision_score(y_teste, y_pred_mlp, average='weighted')
    recall = recall_score(y_teste,y_pred_mlp, average='weighted')
    f1 = f1_score(y_teste, y_pred_mlp, average='weighted')
    
    print("================================================")
    print("DATASET:DRY BEANS | MODELO:MLP")
    print("================================================")
    print("MÉTRICAS:")
    print("\nAcurácia MLP no dry_bean:", acuracia)
    print("\nPrecisão MLP no dry_bean:", precisao)
    print("\nRecall MLP no dry_bean:", recall)
    print("\nF1-Score MLP no dry_bean:", f1)
    
    # Relatório
    print("Relatório MLP dry_bean:\n", classification_report(y_teste, y_pred_mlp))
    
    # Matriz de confusão
    matriz = confusion_matrix(y_teste, y_pred_mlp)
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
    disp.plot(cmap='Blues')
    
    plt.title("Matriz de Confusão - MLP (Dry Bean)")
    plt.show()
    print("\n\n\n\n")
    #Retornando métricas para comparar modelos
    return acuracia, precisao, recall, f1
