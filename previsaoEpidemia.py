import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Gerando dados fictícios para simular uma epidemia
days = 200
time = np.arange(days)
cases = 100 * np.sin(time / 10) + 500 + np.random.normal(scale=20, size=days)  # Simulação básica

df = pd.DataFrame({'day': time, 'cases': cases})

# Normalizando os dados
scaler = MinMaxScaler()
df['cases_normalized'] = scaler.fit_transform(df[['cases']])

# Criando sequências para o modelo
sequence_length = 10
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(df['cases_normalized'].values, sequence_length)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Criando o modelo de previsão baseado em LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Treinando o modelo
model.fit(X, y, epochs=50, batch_size=16, verbose=1)

# Fazendo previsões
y_pred = model.predict(X)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

# Plotando os resultados
plt.figure(figsize=(10, 5))
sns.lineplot(x=df['day'][sequence_length:], y=df['cases'][sequence_length:], label='Casos reais')
sns.lineplot(x=df['day'][sequence_length:], y=y_pred.flatten(), label='Previsão do modelo')
plt.xlabel('Dias')
plt.ylabel('Casos')
plt.legend()
plt.show()
