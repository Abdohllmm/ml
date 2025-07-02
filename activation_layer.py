# Layer d’activation — modifiée légèrement par Zoubir

from layer import Layer  # héritage de la classe de base Layer

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation            # fonction d’activation (ex: tanh, relu)
        self.activation_prime = activation_prime  # dérivée de cette fonction

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)  # application de l’activation
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error  # calcul du gradient
