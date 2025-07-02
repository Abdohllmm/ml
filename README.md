# ml
# 🔬 Réseau de Neurones — Implémentation à la main avec NumPy

Ce projet montre comment créer un petit réseau de neurones **sans utiliser de framework** (comme PyTorch ou TensorFlow), uniquement avec **NumPy**.

## 🧠 Objectif du Projet

Apprendre à coder un réseau de neurones de zéro, comprenant :
- Couches denses (fully connected)
- Couches d’activation (`tanh`)
- Fonction de perte (MSE)
- Propagation avant/arrière (backpropagation)

Ce réseau est entraîné pour résoudre le **problème XOR**.

---

## 🗂️ Structure des fichiers

| Fichier              | Rôle                                                              |
|----------------------|--------------------------------------------------------------------|
| `fc_layer.py`        | Couche entièrement connectée                                      |
| `activation_layer.py`| Couche d’activation (fonction + dérivée)                         |
| `activations.py`     | Fonctions `tanh` et `tanh_prime`                                  |
| `losses.py`          | Fonction de perte MSE + sa dérivée                                |
| `layer.py`           | Classe de base pour les couches                                   |
| `network.py`         | Classe principale `Network` (ajout des couches, fit, predict...)  |
| `main.py`            | Script principal pour entraîner le modèle sur les données XOR     |

---

## ▶️ Exécution

Assure-toi d’avoir Python 3 et NumPy :

```bash
pip install numpy
python main.py
