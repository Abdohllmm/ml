# Fonctions de perte - version personnalisée par Zoubir

import numpy as np

# Erreur quadratique moyenne (MSE)
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

# Dérivée de la fonction MSE
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size
