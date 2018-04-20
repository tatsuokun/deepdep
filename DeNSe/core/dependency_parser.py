import torch
import torch.nn as nn
from torch.autograd import Variable


class DependencyParser(nn.Module):
    def __init__(self,
                 vocab: int,
                 n_pos: int,
                 word_embed_size: int,
                 pos_embed_size: int,
                 hidden_size: int,
                 use_pos: bool,
                 use_cuda: bool):

        super(DependencyParser, self).__init__()
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.pos_embed_size = pos_embed_size if use_pos else 0
        self.bilstm_input_size = word_embed_size + self.pos_embed_size
        self.bilstm_output_size = 2 * hidden_size
        self.use_pos = use_pos
        self.use_cuda = use_cuda

        self.word_emb = nn.Embedding(len(vocab),
                                     word_embed_size,
                                     padding_idx=0)
        self.word_emb.weight.data.copy_(vocab.vectors)
        if self.use_pos:
            self.pos_emb = nn.Embedding(n_pos,
                                        pos_embed_size,
                                        padding_idx=0)
        self.bilstm = nn.LSTM(self.bilstm_input_size,
                              self.hidden_size,
                              num_layers=2,
                              batch_first=True,
                              dropout=0.1,
                              bidirectional=True)
        self.dropout = nn.Dropout(p=0.35)
        self.u_a = nn.Linear(self.bilstm_output_size, self.bilstm_output_size)
        self.w_a = nn.Linear(self.bilstm_output_size, self.bilstm_output_size)
        self.v_a_inv = nn.Linear(self.bilstm_output_size, 1, bias=False)
        self.criterion = nn.NLLLoss(ignore_index=-1)

    def forward(self, input_tokens, input_pos, input_head, phase, output_loss=True):
        loss = 0.0
        batch_size, seq_len = input_tokens.size()
        self.init_hidden(batch_size, use_cuda=self.use_cuda)

        x_i = self.word_emb(input_tokens)
        if self.use_pos:
            pos_embed = self.pos_emb(input_pos)
            x_i = torch.cat((x_i, pos_embed), 2)
        x_i = self.dropout(x_i)

        output, (self.h_n, self.c_n) = self.bilstm(x_i, (self.h_n, self.c_n))
        _, _, hidden_size = output.size()
        parent_prob_table = []
        for i in range(1, seq_len):
            target = output[:, i, :].expand(seq_len, batch_size, -1).transpose(0, 1)
            mask = output.eq(target)[:, :, 0].unsqueeze(2)
            p_head = self.attention(output, target, mask)
            if output_loss:
                loss += self.compute_loss(p_head.squeeze(), input_head[:, i])
            parent_prob_table.append(torch.exp(p_head))

        parent_prob_table = torch.cat((parent_prob_table), dim=2).data
        if self.use_cuda:
            parent_prob_table = parent_prob_table.cpu()
        # tmp = torch.cat((torch.zeros((batch_size, seq_len, 1)), parent_prob_table), dim=2)
        # grandparent_prob_table = torch.bmm(tmp, tmp)[:, 1:]
        _, topi = parent_prob_table.topk(k=1, dim=1)
        preds = topi.squeeze()

        return loss, preds, parent_prob_table

    def compute_loss(self, prob, gold):
        return self.criterion(prob, gold)

    def attention(self, source, target, mask=None):
        function_g = \
            self.v_a_inv(torch.tanh(self.u_a(source) + self.w_a(target)))
        if mask is not None:
            function_g.masked_fill_(mask, -1e3)
        return nn.functional.log_softmax(function_g, dim=1)

    def init_hidden(self, batch_size, use_cuda):
        zeros = Variable(torch.zeros(4, batch_size, self.hidden_size))
        if use_cuda:
            self.h_n = zeros.cuda()
            self.c_n = zeros.cuda()
        else:
            self.h_n = zeros
            self.c_n = zeros
        return self.h_n, self.c_n
