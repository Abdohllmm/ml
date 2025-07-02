# Entraînement d'un petit réseau de neurones sur le problème XOR (version ajustée par Z)

import numpy as np

from model import Network                # anciennement "network.py"
from dense_layer import FCLayer          # anciennement "fc_layer.py"
from activation_block import ActivationLayer  # renommé depuis "activation_layer.py"
from activation_functions import tanh, tanh_prime
from loss_functions import mse, mse_prime

# Données d'entraînement pour XOR
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# Création du réseau
net = Network()
net.add(FCLayer(2, 3))  # Couche d'entrée vers couche cachée
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))  # Couche cachée vers sortie
net.add(ActivationLayer(tanh, tanh_prime))

# Configuration de la fonction de perte
net.use(mse, mse_prime)

# Entraînement du réseau
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# Prédiction finale
out = net.predict(x_train)
print(out)
