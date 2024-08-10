import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Função para carregar as palavras
def carregar_palavras(filepath):
    with open(filepath, 'r') as file:
        palavras = [line.strip() for line in file]
    return palavras

# Função para obter o vetor médio de um comentário
def obter_vetor_medio(comentario, palavras, vetores_palavras):
    palavras_comentario = comentario.split()
    vetores = [vetores_palavras[palavras.index(palavra)] for palavra in palavras_comentario if palavra in palavras]
    if len(vetores) == 0:
        return np.zeros(vetores_palavras.shape[1])
    return np.mean(vetores, axis=0)

# Carregar os arquivos de dados
palavras = carregar_palavras('c:/Users/Felipe/PALAVRAS.txt')
vetores_palavras = np.loadtxt('c:/Users/Felipe/WVECTS.dat')
vetores_textos = np.loadtxt('c:/Users/Felipe/WTEXT.dat')
classes_textos = np.loadtxt('c:/Users/Felipe/CLtx.dat')

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(vetores_textos, classes_textos, test_size=0.2, random_state=42)

# Definir o modelo da rede neural 
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu')) 
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  

# Compilar o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=40, batch_size=10, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia: {accuracy}")
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Novos textos para classificação
novos_textos = [
    "POSITIVO",
    "NEGATIVO",
    "ADOREI O VIDEO",
    "EU NAO GOSTEI",
    "O FILME FOI EXCELENTE",
    "NAO RECOMENDO ESTE PRODUTO",
    "IMPECAVEIS",
    "ESTOU DECEPCIONADO COM A COMPRA",
    "MUITO BOM",
    "NUNCA MAIS VOU COMPRAR AQUI",
    "ESTE FILME E INCRIVEL",
    "O SERVICO FOI HORRIVEL",
    "NAO GOSTEI DO ATENDIMENTO",
    "A COMIDA ESTAVA DELICIOSA",
    "O SERVICO ESTA PIORANDO",
    "VOCES FORAM FANTASTICOS",
    "ODIEI A EXPERIENCIA",
    "ESTOU MUITO SATISFEITO",
    "PESSIMO ATENDIMENTO",
    "EXCELENTE SERVICO",
    "APROVEI O LUGAR",
    "SUPER RECOMENDO",
    "NAO VOLTAREI MAIS",
    "AMEI CADA MOMENTO",
    "NUNCA MAIS USAREI",
    "MELHOR PRODUTO",
    "PIOR SERVICO",
    "ESTAVAM OTIMOS",
    "DECEPCAO TOTAL",
    "EXPERIENCIA PERFEITA"
]

# Vetorizar os novos textos
novos_textos_vetorizados = [obter_vetor_medio(texto, palavras, vetores_palavras) for texto in novos_textos]

# Classificar novos textos
novas_predicoes = (model.predict(np.array(novos_textos_vetorizados)) > 0.5).astype("int32")

# Exibir resultados
for i, texto in enumerate(novos_textos):
    classificacao = "Positivo" if novas_predicoes[i] == 1 else "Negativo"
    print(f"Texto: {texto}")
    print(f"Classificação: {classificacao}\n")

# Função para adicionar novos comentários e classificá-los
def adicionar_novo_comentario():
    while True:
        novo_comentario = input("Digite um novo comentário (ou 'sair' para terminar): ")
        if novo_comentario.lower() == 'sair':
            break
        novo_comentario_vetorizado = obter_vetor_medio(novo_comentario, palavras, vetores_palavras)
        nova_predicao = (model.predict(np.array([novo_comentario_vetorizado])) > 0.5).astype("int32")[0]
        classificacao = "Positivo" if nova_predicao == 1 else "Negativo"
        print(f"Classificação: {classificacao}\n")

# Chamar a função para adicionar novos comentários
adicionar_novo_comentario()
