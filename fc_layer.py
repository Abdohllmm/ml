# Couche entièrement connectée (FC Layer) — version légèrement modifiée par Zoubir

from layer import Layer
import numpy as np

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        # Initialisation aléatoire des poids et biais
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        # Calcul de la sortie : XW + b
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # Calcul des gradients
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # Mise à jour des paramètres (descente de gradient)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        return input_error
