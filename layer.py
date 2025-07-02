# Classe de base Layer — interface commune pour toutes les couches (modif. légère par Z)

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Calcule la sortie Y pour une entrée X donnée
    def forward_propagation(self, input):
        raise NotImplementedError  # À implémenter dans les sous-classes

    # Calcule la dérivée dE/dX à partir de dE/dY
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError  # Doit être défini par chaque couche
