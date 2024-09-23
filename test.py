#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 21:05:57 2024

@author: sounakbhowmik
"""

import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn

# Update the number of qubits to 4 to match the 4 input features
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Update the quantum circuit to work with 4 qubits
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))  # Embed 4 classical data features into 4 qubits
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))  # Entangle the 4 qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]  # Measure expectation values for 4 qubits

# Adjust the number of layers and weight shapes accordingly
n_layers = 3  # Number of layers in the quantum circuit
weight_shapes = {"weights": (n_layers, n_qubits)}

qnode = qml.QNode(quantum_circuit, dev, interface="torch", diff_method="backprop")

# Convert the quantum node to a TorchLayer
quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

# Define a simple hybrid quantum-classical neural network
class HybridNN(nn.Module):
    def __init__(self):
        super(HybridNN, self).__init__()
        self.fc1 = nn.Linear(4, 4)  # Classical layer with 4 input features
        self.qnn = quantum_layer  # Quantum layer with 4 qubits
        self.fc2 = nn.Linear(4, 1)  # Adjust the classical output layer to match the 4 outputs from the quantum layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Classical layer with ReLU activation
        x = self.qnn(x)  # Pass through the quantum layer
        print(x.requires_grad)
        x = self.fc2(x)  # Output layer
        return x

# Example usage:
model = HybridNN()

# Sample input data with 4 features
inputs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])

# Perform a forward pass
output = model(inputs)

#print(output)
