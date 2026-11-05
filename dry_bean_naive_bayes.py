import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


#Lendo a base
df = pd.read_csv(r"C:\Users\roger\Downloads\Dry_Bean.csv")

#Mapeando o gabarito(target) para números 
df['Class'] = df['Class'].map({
    'SEKER': 0,
    'BARBUNYA': 1,
    'BOMBAY': 2,
    'CALI': 3,
    'DERMASON': 4,
    'HOROZ': 5,
    'SIRA': 6
})

#Pegando somente os dados de entrada(features)
x = df[[
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
    "ShapeFactor4"
]]
y = df["Class"]

#Separando os dados de teste dos dados de treino
#teste_size define quantos % da base sera dedica a testes(no caso 20% para testes e 80% para treino)
#random_state define a seed para a aleatorização dos testes
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.2, random_state = 42)

modelo = GaussianNB(var_smoothing=1e-12)

#Treinando o modelo
modelo.fit(x_treino, y_treino)

#Testando o modelo
y_predicao = modelo.predict(x_teste)


# Calculando métricas
print("Acurácia:", accuracy_score(y_teste, y_predicao))
print("Precisão:", precision_score(y_teste, y_predicao, average='weighted'))
print("Recall:", recall_score(y_teste, y_predicao, average='weighted'))
print("F1-Score:", f1_score(y_teste, y_predicao, average='weighted'))

# Relatório completo por classe
print("\nRelatório de classificação:")
print(classification_report(y_teste, y_predicao))

# Plotando a matriz de confusão
matriz = confusion_matrix(y_teste, y_predicao)

disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
disp.plot(cmap='Blues')

plt.title("Matriz de Confusão - Naive Bayes")
plt.show()