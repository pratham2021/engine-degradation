import torch
import torch.nn as nn

class LSTMModel(nn.Module):
  def __init__(self, inputSize=58, hiddenSize=64, numLayers=2, batchFirst=True, dropOut=0.2):
    super().__init__()
    self.lstm_layer = nn.LSTM(inputSize, hiddenSize, numLayers, batch_first=True, dropout=dropOut)
    self.linear_layer = nn.Linear(in_features=hiddenSize, out_features=1)
    
  def forward(self, x):
    output, _ = self.lstm_layer(x)
    return self.linear_layer(output[:, -1, :]).squeeze()