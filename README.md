# Classificação de Comentários com Redes Neurais

##  Descrição do Projeto

Este projeto tem como objetivo o desenvolvimento de um modelo de **classificação de comentários** utilizando **Redes Neurais Artificiais**, classificando textos como **positivos** ou **negativos**.

A solução utiliza técnicas de **Processamento de Linguagem Natural (PLN)** combinadas com **Aprendizado de Máquina**, aplicando vetores de palavras pré-treinados e uma rede neural do tipo **MLP (Multi-Layer Perceptron)**.

---

##  Tecnologias Utilizadas

* **Linguagem:** Python
* **Bibliotecas:**

  * TensorFlow / Keras
  * NumPy
  * scikit-learn
* **Dados:**

  * Vetores de palavras pré-treinados
  * Comentários previamente anotados

---

##  Metodologia

### Extração de Características

Cada comentário é convertido em um **vetor médio**, calculado a partir da média dos vetores das palavras presentes no texto. Palavras não encontradas no vocabulário são ignoradas. Caso nenhuma palavra seja reconhecida, o comentário é representado por um vetor nulo.

### Modelo de Classificação

Foi adotada uma **Rede Neural Densa (MLP)**. Diferentes arquiteturas foram testadas:

* 3 camadas ocultas (64 / 32 / 8 neurônios)
* 2 camadas ocultas (8 / 4 neurônios)
* 2 camadas ocultas (32 / 16 neurônios)

A arquitetura com **32 e 16 neurônios** apresentou melhor desempenho e foi escolhida para o modelo final.

### Configuração do Treinamento

* **Divisão dos dados:** 80% treino / 20% teste
* **Função de ativação:** ReLU (camadas ocultas) e Sigmoid (camada de saída)
* **Função de perda:** Binary Crossentropy
* **Otimizador:** Adam
* **Early Stopping:** aplicado para evitar overfitting

### Avaliação

O modelo é avaliado utilizando:

* Acurácia
* Perda (loss)
* Relatório de classificação

---

##  Execução do Projeto

### Pré-requisitos

Certifique-se de ter o Python instalado e as dependências necessárias:

```bash
pip install numpy tensorflow scikit-learn
```

### Estrutura Esperada de Arquivos

O projeto utiliza os seguintes arquivos de dados:

* `PALAVRAS.txt` – Lista de palavras do vocabulário
* `WVECTS.dat` – Vetores das palavras
* `WTEXT.dat` – Vetores médios dos textos
* `CLtx.dat` – Classes dos textos

> ⚠️ Os caminhos dos arquivos devem ser ajustados conforme o ambiente local.

### Execução

Execute o script principal em Python. Após o treinamento, o modelo:

1. Avalia o desempenho no conjunto de teste
2. Classifica uma lista de novos comentários
3. Permite a inserção de comentários via terminal para classificação em tempo real

---

##  Exemplo de Uso

O sistema classifica comentários como:

* "O filme foi excelente" → **Positivo**
* "Não gostei do atendimento" → **Negativo**

Também é possível digitar novos comentários diretamente no terminal e obter a classificação instantaneamente.

---

##  Resultados

O modelo apresentou **acurácia satisfatória**, classificando corretamente cerca de 80% dos novos comentários, demonstrando a eficiência do uso de redes neurais na tarefa de análise de sentimentos. A estratégia de vetores médios mostrou-se simples e eficaz, embora limitada por palavras fora do vocabulário.

---

##  Conclusão

Este projeto demonstra uma aplicação prática de **Inteligência Artificial e PLN** para classificação de sentimentos em textos. A solução desenvolvida atende aos objetivos propostos e serve como base para futuras melhorias, como o uso de embeddings mais robustos ou modelos mais avançados.

---

##  Autor

**Felipe Filla Vitorino**
Curso de Tecnologia em Análise e Desenvolvimento de Sistemas – UFPR
