
import torch
from Model import QCNN

qcnn = QCNN()


x = torch.rand(10,45)
y = qcnn(x)
#print(y.requires_grad)