from torch import nn
from torch.nn import functional as F
import torch


class Rnn(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_classes, num_layers=2, dropout_prob=0.2, classification=True, rnn_type='LSTM', final_layer='fc'):
        super(Rnn, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.classification = classification
        self.final_layer = final_layer
        self.rnn_type = rnn_type
        self.dropout_prob = dropout_prob

        if self.rnn_type == 'LSTM':

            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=self.dropout_prob)
            self.rnn1 = nn.LSTM(input_size=hidden_size, hidden_size=num_classes, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
            self.rnn1 = nn.GRU(input_size=hidden_size, hidden_size=num_classes, num_layers=num_layers,
                               batch_first=True)
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.fc = nn.Linear(hidden_size,  hidden_size//2 if self.final_layer == 'fc1' else num_classes)
        if self.final_layer == 'fc1':
            self.fc1 = nn.Linear(hidden_size//2, num_classes)

    def forward(self, x, hc):
        # the order must be batch, seq_len, input size

        if hc is None:
            out, (ht, ct) = self.rnn(x)
        else:
            out, (ht, ct) = self.rnn(x, (hc[0], hc[1]))

        if self.final_layer == 'fc':
            if self.classification:
                pass

            out = self.dropout(out)
            out = self.fc(out)

        elif self.final_layer == 'fc1':
            if self.classification:
                out = out[:, -1]

            out = self.dropout(out)
            out = self.fc(out)
            out = F.relu(out)
            out = self.dropout(out)
            out = self.fc1(out)

        elif self.final_layer == 'lstm':
            out, (ht, ct) = self.rnn1(out)
            if self.classification:
                out = out[:, -1]

        return out, ht, ct

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)


