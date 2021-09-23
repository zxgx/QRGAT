import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional, num_layers):
        super(EncoderRNN, self).__init__()
        if bidirectional and hidden_size % 2 != 0:
            raise RuntimeError('hidden_size is expected to be even in the bidirectional mode!')
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.model = nn.LSTM(input_size, self.hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """x: [batch_size * max_length * emb_dim]
           x_len: [batch_size]
        """
        sorted_x_len, indx = torch.sort(x_len, 0, descending=True)
        x = pack_padded_sequence(x[indx], sorted_x_len, batch_first=True)

        packed_h, (packed_h_t, packed_c_t) = self.model(x)

        if self.num_directions == 2:
            packed_h_t = torch.cat((packed_h_t[-2], packed_h_t[-1]), 1)  # batch size, hidden dim
            packed_c_t = torch.cat((packed_c_t[-2], packed_c_t[-1]), 1)  # batch size, hidden dim
        else:
            packed_h_t = packed_h_t[-1]
            packed_c_t = packed_c_t[-1]

        # restore the sorting
        inverse_indx = torch.argsort(indx, 0)

        hh, _ = pad_packed_sequence(packed_h, batch_first=True)
        restore_hh = hh[inverse_indx]  # batch size, max seq len, hidden dim
        # restore_hh = restore_hh.transpose(0, 1)  # max seq len, batch size, hidden dim

        restore_packed_h_t = packed_h_t[inverse_indx]
        # restore_packed_h_t = restore_packed_h_t.unsqueeze(0) # [1, batch_size, emb_dim]

        restore_packed_c_t = packed_c_t[inverse_indx]
        # restore_packed_c_t = restore_packed_c_t.unsqueeze(0) # [1, batch_size, emb_dim]
        rnn_state_t = (restore_packed_h_t, restore_packed_c_t)

        return restore_hh, rnn_state_t
