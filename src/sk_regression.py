import numpy as np
import pandas as pd
from skimage import io, color
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Charger l'image
image_path = '/mer.jpg'  # Remplace par ton image
image_color = io.imread(image_path)  # Charger l'image couleur
image_gray = color.rgb2gray(image_color)  # Convertir en noir et blanc

# Dimensions de l'image
h, w, _ = image_color.shape

# Préparer les données pour l'entraînement
data = []
target = []

for x in range(h):
    for y in range(w):
        luminance = image_gray[x, y]
        r, g, b = image_color[x, y]
        data.append([x, y, luminance])
        target.append([r, g, b])

data = np.array(data)
target = np.array(target)

# Diviser les données en un petit sous-ensemble pour accélérer l'entraînement
sample_size = 1000  # Prendre un sous-ensemble aléatoire
indices = np.random.choice(len(data), sample_size, replace=False)
data_sample = data[indices]
target_sample = target[indices]

# Entraîner un modèle simple de régression linéaire
model = LinearRegression()
model.fit(data_sample, target_sample)

# Prédire les couleurs pour chaque pixel de l'image
predicted_colors = model.predict(data)

# Reconstruction de l'image colorisée
image_pred = np.clip(predicted_colors.reshape(h, w, 3), 0, 255).astype(np.uint8)

# Sauvegarder et afficher les résultats
io.imsave('image_colorized_simple.jpg', image_pred)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_color)
axes[0].set_title('Image Originale')

axes[1].imshow(image_gray, cmap='gray')
axes[1].set_title('Image Noir et Blanc')

axes[2].imshow(image_pred)
axes[2].set_title('Image Colorisée')

plt.show()