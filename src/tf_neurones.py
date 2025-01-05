import sys
import numpy as np
from skimage import io, color
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Augmenter la limite de récursion
sys.setrecursionlimit(5000)

# Charger et redimensionner l'image
image_path = 'mer.jfif'  # Remplace par ton image

image_color = io.imread(image_path)
image_color = resize(image_color, (256, 256), anti_aliasing=True)  # Redimensionne à 256x256
image_color = (image_color * 255).astype(np.uint8)  # Reconvertir en entier
image_gray = color.rgb2gray(image_color)

# Dimensions de l'image
h, w, _ = image_color.shape

# Préparer les données pour l'entraînement
data = []
target = []

for x in range(h):
    for y in range(w):
        luminance = image_gray[x, y]
        r, g, b = image_color[x, y]
        data.append([x / h, y / w, luminance])  # Normalisation des coordonnées
        target.append([r / 255, g / 255, b / 255])  # Normalisation des couleurs

data = np.array(data)
target = np.array(target)

# Échantillonnage des données
sample_size = 5000  # Réduit le nombre d'échantillons
indices = np.random.choice(len(data), sample_size, replace=False)
data_sample = data[indices]
target_sample = target[indices]

# Diviser les données en entraînement et validation
split_index = int(len(data_sample) * 0.8)
X_train, X_val = data_sample[:split_index], data_sample[split_index:]
y_train, y_val = target_sample[:split_index], target_sample[split_index:]

# Construire un réseau de neurones simple
model = Sequential([
    Input(shape=(3,)),
    Dense(32, activation='relu'),  # Réduction de la complexité du modèle
    Dense(64, activation='relu'),
    Dense(3, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entraîner le modèle
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=256)

# Prédire les couleurs pour chaque pixel
predicted_colors = model.predict(data, batch_size=512)

# Reconstruction de l'image colorisée
image_pred = (predicted_colors.reshape(h, w, 3) * 255).astype(np.uint8)

# Sauvegarder et afficher les résultats
io.imsave('image_colorized_nn_resized.jpg', image_pred)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_color)
axes[0].set_title('Image Originale')

axes[1].imshow(image_gray, cmap='gray')
axes[1].set_title('Image Noir et Blanc')

axes[2].imshow(image_pred)
axes[2].set_title('Image Colorisée')

plt.show()