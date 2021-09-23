import torch
import torch.nn as nn
import torch.nn.functional as F


class Instruction(nn.Module):
    def __init__(self, word_dim, hidden_dim, question_dropout, linear_dropout, num_step):
        super(Instruction, self).__init__()

        self.lstm = nn.LSTM(word_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.hidden_dim = hidden_dim

        self.question_dropout = nn.Dropout(question_dropout)
        self.linear_dropout = nn.Dropout(linear_dropout)
        self.num_step = num_step
        for i in range(num_step):
            self.add_module('question_linear'+str(i), nn.Linear(hidden_dim, hidden_dim))

        self.inst_ques_linear = nn.Linear(hidden_dim*2, hidden_dim)
        self.attention_linear = nn.Linear(hidden_dim, 1)

    def forward(self, question, question_mask):
        """

        :param question: [ batch size, max seq len, word dim ]
        :param question_mask: [ batch size, max seq len ]
        :return:
        """
        # batch size, max seq len, word dim
        question = self.question_dropout(question)

        max_seq_len = question_mask.shape[1]

        question_len = question_mask.long().sum(dim=1)  # batch size
        question_len, idx = torch.sort(question_len, descending=True)

        packed_input = nn.utils.rnn.pack_padded_sequence(question[idx], question_len, batch_first=True)
        packed_output, (hidden, _) = self.lstm(packed_input)
        question_token, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=max_seq_len)

        recover_idx = torch.argsort(idx)
        question_token = question_token[recover_idx]  # batch size, max seq len, hidden dim
        hidden = hidden.squeeze(0)[recover_idx].unsqueeze(1)  # batch size, 1, hidden dim

        instructions, attentions = [], []
        instruction = torch.zeros_like(hidden)  # batch size, 1, hidden dim
        for i in range(self.num_step):
            question_linear = getattr(self, 'question_linear'+str(i))
            question_i = question_linear(self.linear_dropout(hidden))  # batch size, 1, hidden dim
            # batch size, 1, hidden dim
            inst_ques = self.inst_ques_linear(self.linear_dropout(torch.cat([instruction, question_i], dim=2)))
            # batch size, max seq len
            attention = self.attention_linear(self.linear_dropout(inst_ques * question_token)).squeeze(2)
            attention = question_mask * attention + (1 - question_mask) * -1e20
            attention = F.softmax(attention, dim=1)  # batch size, max seq len
            # batch size, hidden dim
            instruction = torch.sum(attention.unsqueeze(2) * question_token, dim=1)

            instructions.append(instruction)
            attentions.append(attention)
            instruction = instruction.unsqueeze(1)
        return instructions, hidden, attentions