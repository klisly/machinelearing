from ctypes import pointer
from re import S
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
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim / 2 if bidir else hidden_dim
        self.n_layers = n_layers * 2 if bidir else n_layers
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, 
            n_layers, dropout = dropout, bidirectional = bidirectional)
        self.h0 = Parameter(torch.zeros(1), requires_grad=True)
        self.c0 = Parameter(torch.zeros(1), requires_grad=True)
        
    def forward(self, embedded_inputs, hidden):
        embedded_inputs = embedded_inputs.permute(1, 0, 2)
        output, hidden = self.lstm(embedded_inputs, hidden)
        return output.permute(1, 0, 2)
    
    def init_hidden(self, embedded_inputs):
        batch_size = embedded_inputs.size(0)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers, batch_size, self.hidden_dim)
        c0 = self.c0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers, batch_size, self.hidden_dim)
        return h0, c0
    
class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Linear(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad = True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad = False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        nn.init.uniform(self.V, -1, 1)
    
    def forward(self, input, context, mask):
        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context,)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]

        alpha = self.softmax(att)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha
    
    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)



class Decoder(nn.Modules):
    def __init__(self, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(embedding_dim * 2,  hidden_dim)

        self.att = Attention(hidden_dim, hidden_dim)

        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs, decoder_input, hidden, context):
        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask)

        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = 1
        
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()
        outputs = []
        pointers = []

        def step(x, hidden):
            h, c = hidden
            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4,1)
            input = F.sigmoid(input)
            forget = F.sigmoid(forget)
            cell = F.tanh(cell)
            out = F.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * F.tanh(c_t)

            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = F.tanh(self.hidden_out(torch.cat(hidden_t, h_t), 1))
            return hidden_t, c_t, output
        
        for _ in range(input_length):
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            masked_outs = outs * mask
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            mask = mask * (1 - one_hot_pointers)

            embedded_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedded_dim).byte()
            decoder_input = embedded_inputs[embedded_mask.data].view(batch_size, self.embedding_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))
        
        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)
        return (outputs, pointers), hidden

class PointerNet(nn.Module):
    def __init__(self,
        embedded_dim, hidden_dim, lstm_layers, dropout, bidir=False
        ) -> None:
        super().__init__()  
        self.embedded_dim = embedded_dim
        self.bidir = bidir 
        self.embedding = nn.Linear(2, embedded_dim)
        self.encoder = Encoder(embedded_dim, hidden_dim, lstm_layers, dropout, bidir)
        self.decoder = Decoder(embedded_dim, hidden_dim)

        self.decoder_input0 = Parameter(torch.FloatTensor(embedded_dim), requires_grad=False)
        nn.init.uniform(self.decoder_input0, -1, 1)
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        inputs = inputs.view(batch_size * input_length, -1)
        embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)

        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs, encoder_hidden0)
        if self.bidir:
            decoder_hidden0 = (torch.cat(encoder_hidden[0][-2:], dim = -1),torch.cat(encoder_hidden[1][-2:], dim = -1))
        else:
            encoder_hidden0 = (encoder_hidden[0][-1], encoder_hidden[1][-1])
        (outputs, pointers), decoder_hidden = self.decoder(embedded_inputs, decoder_input0, decoder_hidden0, encoder_outputs)

        return outputs, pointers