from unicodedata import bidirectional
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F 

class Encoder(nn.Module):
    """
    Encoder class for Pointer Net
    """
    def __init__(self, embedding_dim:int, hidden_dim:int,
        n_layers:int, dropout:float, bidir:bool) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim / 2 if bidir else hidden_dim
        self.n_layers = n_layers * 2 if bidir else n_layers
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, 
            n_layers, dropout = dropout, bidirectional = bidirectional)
        self.h0 = Parameter(torch.zeros(1), requires_grad=True)
        self.c0 = Parameter(torch.zeros(1), requires_grad=True)
        