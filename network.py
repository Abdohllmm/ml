# Classe Network — structure de base d’un réseau de neurones (ajustée par Z)

class Network:
    def __init__(self):
        self.layers = []          # liste des couches ajoutées
        self.loss = None          # fonction de perte
        self.loss_prime = None    # dérivée de la fonction de perte

    def add(self, layer):
        self.layers.append(layer)  # ajout d’une couche au réseau

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]

                # propagation avant à travers toutes les couches
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # accumulation de l’erreur
                err += self.loss(y_train[j], output)

                # rétropropagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
