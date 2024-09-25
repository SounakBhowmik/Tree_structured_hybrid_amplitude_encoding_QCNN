#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:19:18 2024

@author: sounakbhowmik
"""
from pennylane.operation import Operation
from pennylane import numpy as np
import torch.nn as nn

import pennylane as qml
import torch
from torch.nn import Module


import math

n_qubits = 12

def call_control_values(m, n):
    res = [bool(int(x)) for x in bin(n).split('b')[-1]]
    aux = [False]*(m-len(res))
    return aux + res

def tree_structured_hybrid_amplitude_sub_encode(n:int, start:int, x:torch.tensor):
    i = 0
    for i in range(n):
        for j in range(2**i):
            if (i==0):
                qml.RY(x[2**i + j -1], wires=i+start)
            else:
                qml.ctrl(qml.RY, control = range(start, i+start), control_values = call_control_values(i,j))(x[2**i + j -1], wires = i+start)
            
    #return [qml.expval(qml.PauliZ(wires = i )) for i in range(n)]


def tree_structured_hybrid_amplitude_encode(x:torch.tensor, m:int, n:int=None):
    feature_len = len(x)
    if(n == None):
        n = (int)(feature_len / m)
        n = math.log(n + 1)
    assert feature_len == m * (2**n - 1)
    assert m*n == n_qubits
    
    grp_size = (int)(feature_len/m)
    groups = torch.split(x, grp_size)
    for w, group_data in enumerate(groups):
        tree_structured_hybrid_amplitude_sub_encode(n=n, start=w*n, x = group_data)
            
    #return [qml.expval(qml.PauliZ(wires = i )) for i in range(n_qubits)]


#print(qml.draw_mpl(tree_structured_hybrid_amplitude_encode)(n_qubits=12, x=torch.rand(45), m=3, n=4))
'''
#########################################################################################

class Tree_structured_hybrid_amplitude_encode(Operation):
    num_wires = qml.operation.AnyWires  # Allow encoding on any number of qubits
    grad_method = None  # Set gradient method to None if it's not differentiable

    def __init__(self, features, wires, m, n, id=None):
        # Initialize the features and wires
        super().__init__(features, wires=wires, id=id)

        # Ensure features have a batch dimension
        self.features = features
        #self.wires = wires
        self.m = m
        self.n = n
    
    @staticmethod
    def call_control_values(m, n):
        res = [bool(int(x)) for x in bin(n).split('b')[-1]]
        aux = [False]*(m-len(res))
        return aux + res
    @staticmethod
    def tree_structured_hybrid_amplitude_sub_encode(n:int, start:int, x:torch.tensor):
        i = 0
        for i in range(n):
            for j in range(2**i):
                if (i==0):
                    qml.RY(x[2**i + j -1], wires=i+start)
                else:
                    qml.ctrl(qml.RY, control = range(start, i+start), control_values = call_control_values(i,j))(x[2**i + j -1], wires = i+start)
                    
    def tree_structured_hybrid_amplitude_encode(self, x:torch.tensor):
        m:int = self.m
        n:int = self.n
        feature_len = len(x)
        
        if(n == None):
            n = (int)(feature_len / m)
            n = math.log(n + 1)
        else:
            n = self.n
        
        ####
        assert feature_len == m * (2**n - 1)
        # assert m*n == n_qubits
        
        grp_size = (int)(feature_len/m)
        print(x.shape)
        groups = torch.split(x, grp_size)
        for w, group_data in enumerate(groups):
            self.tree_structured_hybrid_amplitude_sub_encode(n=n, start=w*n, x = group_data)
            
            
    def apply_encoding(self, features):
        """Creates the quantum operations for a single set of features."""
        ops = []
        for _, feature in enumerate(features):
            print(type(feature))
            ops.append(self.tree_structured_hybrid_amplitude_encode(features))
        return ops

    def decomposition(self):
        """Provides a decomposition of the operation as a list of simpler operations."""
        ops = []
        for feature_set in self.features:
            ops.extend(self.apply_encoding(feature_set))
        return ops




#########################################################################################
'''


#@qml.qnode(qml.device('default.qubit'))
def conv(wires:list, W:torch.tensor):
    
    assert len(wires) == 2
    
    qml.U3(W[0], W[1], W[2], wires = wires[0])
    qml.U3(W[3], W[4], W[5], wires = wires[1])
        
    qml.CNOT(wires=wires)
    
    qml.RY(W[6], wires=wires[0])
    qml.RZ(W[7], wires=wires[1])
    
    qml.CNOT(wires= list(reversed(wires)))
    
    qml.RY(W[8], wires=wires[0])
    
    qml.CNOT(wires=wires)
    
    
    qml.U3(W[9], W[10], W[11], wires = wires[0])
    qml.U3(W[12], W[13], W[14], wires = wires[1])
    
    #return [qml.expval(qml.PauliZ(wires = i )) for i in wires]


#print(qml.draw_mpl(conv)(wires=[11,0], W=torch.rand(15)))

#@qml.qnode(qml.device('default.qubit'))
def pool(wires:list,  W:torch.tensor):
    assert len(wires) == 2

        
    qml.CRZ(phi=W[0], wires=wires)
    
    qml.X(wires=wires[0])
    
    qml.CRZ(phi=W[1], wires=wires)
    #return [qml.expval(qml.PauliZ(wires = i )) for i in wires]


        
        
#print(qml.draw_mpl(pool)(wires=[0,1], W=torch.rand(2)))

# The QCNN model

class QCNN(Module):
    def __init__(self, in_features:int = 45, n_layers:int = 3, encode_m = 3, encode_n=4, device = None, dtype=None):
        super().__init__()
        
        assert in_features == encode_m*(2**encode_n -1)
        
        self.in_features = in_features
        self.n_layers = n_layers
        self.device = (qml.device('lightning.gpu', wires=self.in_features) if torch.cuda.is_available() else qml.device('default.qubit', wires=self.in_features)) if device == None else device
        self.encode_m = encode_m
        self.encode_n = encode_n
        self.n_wires  = encode_m * encode_n
        self.wires_structure = [
                                [
                                    [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11]],
                                    [[1,2], [3,4], [5,6], [7,8], [9,10], [11,0]],
                                    [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11]]
                                ],
                                
                                [
                                    [[1,3], [5,7], [9, 11]],
                                    [[2,4], [6,8], [10, 0]],
                                    [[1,3], [5,7], [9, 11]]
                                ],
                                
                                [
                                    [[3,7]], 
                                    [[3,7]]
                                ],
                                
                                [
                                    [[7,11]], 
                                    [[7,11]]
                                ]
                                ]
        self.weight_shapes = {"weights10": (2, 6, 15), "weights11": (1,6,2),
                         "weights20": (2, 3, 15), "weights21": (1,3,2),
                         "weights30": (1, 1, 15), "weights31": (1,1,2),
                         "weights40": (1, 1, 15), "weights41": (1,1,15)}
        
        self.qnode = qml.QNode(self.quantum_circuit, self.device, interface="torch", diff_method="backprop")
        self.qlayer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes)
        
    def quantum_circuit(self, inputs, weights10, weights11, weights20, weights21, weights30, weights31, weights40, weights41):

        #encoding
        tree_structured_hybrid_amplitude_encode(inputs, m = self.encode_m, n=self.encode_n)
        
        
        weights_array = [weights10, weights11, weights20, weights21, weights30, weights31, weights40, weights41]
        
        for layer in range(len(self.wires_structure)):
            #print(layer)
            # implement 2 conv layers
            conv_weights = weights_array[layer*2]
            w = 0
            for i in range(conv_weights.shape[0]):
                w = i
                for j in range(conv_weights.shape[1]):
                    #print(self.wires_structure[layer][i][j])
                    conv(wires = self.wires_structure[layer][i][j], W=conv_weights[i,j])
            w +=1
            pool_weights = weights_array[layer*2 + 1]
            for i in range(pool_weights.shape[0]):
                for j in range(pool_weights.shape[1]):
                    #print(self.wires_structure[layer][w][j])
                    pool(wires = self.wires_structure[layer][w][j], W=pool_weights[i,j])
               
        return qml.expval(qml.PauliZ(wires=self.n_wires-1))
    
    def forward(self,inputs):
        inputs = inputs.detach()
        ret = torch.empty(0, dtype= torch.float32)
        for x in inputs:
            y = torch.unsqueeze(torch.sigmoid(self.qlayer(x)), 0)
            #print(y.shape)
            assert y.shape == torch.Size([1])
            ret = torch.cat((ret, y))
        #assert ret.requires_grad == True
        return torch.unsqueeze(ret, 1)
'''           
a = torch.rand(2, 45)
qcnn = QCNN()
qcnn(a)            
'''

