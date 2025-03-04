import torch
from torch import nn
from model import NeuralNetwork

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
prediction_probabilities = nn.Softmax(dim=1)(logits)
y_prediction = prediction_probabilities.argmax(1)
print(y_prediction)
