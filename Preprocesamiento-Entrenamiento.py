import os
import gc
import json
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, LeakyReLU, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

data0D = pd.read_csv('/kaggle/input/zipfile/0D.csv')
data0E = pd.read_csv('/kaggle/input/zipfile/0E.csv')
data1D = pd.read_csv('/kaggle/input/zipfile/1D.csv')
data1E = pd.read_csv('/kaggle/input/zipfile/1E.csv')
data2D = pd.read_csv('/kaggle/input/zipfile/2D.csv')
data2E = pd.read_csv('/kaggle/input/zipfile/2E.csv')
data3D = pd.read_csv('/kaggle/input/zipfile/3D.csv')
data3E = pd.read_csv('/kaggle/input/zipfile/3E.csv')
data4D = pd.read_csv('/kaggle/input/zipfile/4D.csv')
data4E = pd.read_csv('/kaggle/input/zipfile/4E.csv')

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Configuration
labels = {'no_unbalance': 0, 'unbalance': 1}
sensors = ['Vibration_1', 'Vibration_2', 'Vibration_3']
samples_per_second = 4096
seconds_per_analysis = 1.0
window = int(samples_per_second * seconds_per_analysis)

# Feature extraction function
def get_features(data, label):
    n = int(np.floor(len(data) / window))
    data = data[:int(n) * window]
    X = data.values.reshape((n, window))
    y = np.ones(n) * labels[label]
    return X, y

# Extract data for all sensors
def prepare_data(data, sensors, label):
    sensor_data = []
    for sensor in sensors:
        X_sensor, _ = get_features(data[sensor], label)
        sensor_data.append(X_sensor)
    return np.concatenate(sensor_data, axis=1)

# Prepare training and validation datasets
X0 = prepare_data(data0D, sensors, "no_unbalance")
X1 = prepare_data(data1D, sensors, "unbalance")
X2 = prepare_data(data2D, sensors, "unbalance")
X3 = prepare_data(data3D, sensors, "unbalance")
X4 = prepare_data(data4D, sensors, "unbalance")
X = np.concatenate([X0, X1, X2, X3, X4])
y = np.concatenate([np.zeros(X0.shape[0]), np.ones(X1.shape[0] + X2.shape[0] + X3.shape[0] + X4.shape[0])])

X0_val = prepare_data(data0E, sensors, "no_unbalance")
X1_val = prepare_data(data1E, sensors, "unbalance")
X2_val = prepare_data(data2E, sensors, "unbalance")
X3_val = prepare_data(data3E, sensors, "unbalance")
X4_val = prepare_data(data4E, sensors, "unbalance")
X_val = np.concatenate([X0_val, X1_val, X2_val, X3_val, X4_val])
y_val = np.concatenate([np.zeros(X0_val.shape[0]), np.ones(X1_val.shape[0] + X2_val.shape[0] + X3_val.shape[0] + X4_val.shape[0])])

# Train-test split
train_test_ratio = 0.9
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_test_ratio, random_state=42)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Apply FFT
X_train_fft = np.abs(np.fft.rfft(X_train, axis=1))[:, :int(window / 2)]
X_test_fft = np.abs(np.fft.rfft(X_test, axis=1))[:, :int(window / 2)]
X_val_fft = np.abs(np.fft.rfft(X_val, axis=1))[:, :int(window / 2)]

# Remove DC component
X_train_fft[:, 0] = 0
X_test_fft[:, 0] = 0
X_val_fft[:, 0] = 0

# Normalize the data
scaler = RobustScaler(quantile_range=(5, 95))
X_train_fft_sc = scaler.fit_transform(X_train_fft)
X_test_fft_sc = scaler.transform(X_test_fft)
X_val_fft_sc = scaler.transform(X_val_fft)

# Split validation data into subsets
split_val = len(X_val_fft_sc) // 5
X_val_subsets = [X_val_fft_sc[i * split_val:(i + 1) * split_val] for i in range(5)]
y_val_subsets = [y_val[i * split_val:(i + 1) * split_val] for i in range(5)]

# Train and evaluate models
accuracies_per_class = []

for i in range(5):  # Train 5 models with different numbers of hidden layers
    X_in = Input(shape=(X_train_fft_sc.shape[1],))
    x = X_in
    
    for _ in range(i):  # Add hidden layers according to the iteration
        x = Dense(units=1024, activation='linear')(x)
        x = LeakyReLU(alpha=0.05)(x)
    
    X_out = Dense(units=1, activation='sigmoid')(x)
    model = Model(inputs=X_in, outputs=X_out)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint_path = f"model_with_{i}_hidden_layers.keras"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit(X_train_fft_sc, y_train, epochs=100, batch_size=128, validation_data=(X_test_fft_sc, y_test), callbacks=[checkpoint])
    model = load_model(checkpoint_path)
    
    val_accs = []
    for X, y in zip(X_val_subsets, y_val_subsets):
        val_acc = model.evaluate(X, y, verbose=0)[1]  # Accuracy
        val_accs.append(val_acc)
    
    accuracies_per_class.append(val_accs)

# Display results
import pandas as pd
df_accuracies = pd.DataFrame(accuracies_per_class, columns=[f'Class {i + 1}' for i in range(5)])
df_accuracies['Average'] = df_accuracies.mean(axis=1)
print("\n Accuracy for each model by class")
print(df_accuracies)

import matplotlib.pyplot as plt
import numpy as np

# Recuperar los datos generados en tu código
# Usa directamente el DataFrame `df_accuracies` generado por tu código
unbalance_factors = [0, 40, 80, 120, 140]  # Factores de desequilibrio
labels = ["0E", "1E", "2E", "3E", "4E"]  # Etiquetas de las clases

# Convertimos el DataFrame a una lista para usar en la gráfica
accuracies_per_class = df_accuracies.iloc[:, :-1].values.tolist()  # Excluir columna de promedio

# Crear la gráfica
plt.figure(figsize=(8, 6))

colors = ['blue', 'orange', 'green', 'red', 'purple']
markers = ['+', 'x', 'o', 's', 'd']

# Iterar sobre los datos de cada modelo para crear la gráfica
for i, acc in enumerate(accuracies_per_class):
    mean_acc = np.mean(acc) * 100  # Calcular la media en porcentaje
    plt.plot(
        unbalance_factors,
        acc,
        label=f"{i} hidden FC layers, mean: {mean_acc:.1f}%",
        marker=markers[i],
        linestyle="--",
        color=colors[i]
    )

# Configuración de la gráfica
plt.title("(e) Precisión en datasets de Validación")
plt.xlabel("Factor de desequilibrio [mm g]")
plt.ylabel("Precisión")
plt.xticks(unbalance_factors, labels)
plt.ylim([0.4, 1.05])  # Ajustar el rango del eje Y
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()

# Mostrar la gráfica
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Datos de precisiones de cada modelo (accuracies_per_class) ya calculados
# Precisión para cada par de clases (0E vs 1E, 0E vs 2E, etc.)
accuracies_per_class = [
    [0.856, 0.870, 0.880, 0.991],  # 0 hidden layers
    [0.887, 0.892, 0.901, 0.998],  # 1 hidden layer
    [0.889, 0.900, 0.920, 1.000],  # 2 hidden layers
    [0.851, 0.865, 0.878, 0.995],  # 3 hidden layers
    [0.868, 0.882, 0.892, 0.999],  # 4 hidden layers
]

# Factores de desequilibrio correspondientes a cada par
unbalance_factors = [60, 80, 100, 140]  # Factores de desequilibrio mm·g

# Precisión promedio por configuración
means = [np.mean(acc) for acc in accuracies_per_class]

# Crear la gráfica
plt.figure(figsize=(10, 6))

# Iterar sobre los datos de cada modelo para crear la gráfica
for i, acc in enumerate(accuracies_per_class):
    mean_acc = np.mean(acc) * 100  # Multiplicar por 100 para obtener porcentaje
    plt.plot(
        unbalance_factors,
        acc,
        label=f"{i} hidden FC layers, mean: {mean_acc:.1f}%",  # Mostrar como porcentaje
        marker=markers[i],
        linestyle="--",
        color=colors[i]
    )

# Configuración del gráfico
plt.title('Precisión por pares (0E vs otros)', fontsize=14)
plt.xlabel('Factor de desequilibrio [mm g]', fontsize=12)
plt.ylabel('Precisión', fontsize=12)
plt.ylim(0.5, 1.05)  # Ajustar el rango del eje Y
plt.xticks(unbalance_factors, labels=['0E+1E', '0E+2E', '0E+3E', '0E+4E'])  # Etiquetas de pares
plt.legend(title="Model Configuration", loc='lower right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Mostrar la gráfica
plt.tight_layout()
plt.show()