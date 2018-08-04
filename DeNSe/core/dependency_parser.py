import torch
import torch.nn as nn
from allennlp.modules.elmo import batch_to_ids, Elmo


class DependencyParser(nn.Module):
    def __init__(self,
                 vocab,
                 pos,
                 word_embed_size: int,
                 pos_embed_size: int,
                 hidden_size: int,
                 use_pos: bool,
                 use_elmo: bool,
                 use_cuda: bool,
                 inference: bool):

        super(DependencyParser, self).__init__()
        self.vocab_size = len(vocab)
        self.pos_size = len(pos)
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.pos_embed_size = pos_embed_size if use_pos else 0
        self.bilstm_input_size = word_embed_size + self.pos_embed_size
        self.bilstm_output_size = 2 * hidden_size
        self.use_pos = use_pos
        self.use_cuda = use_cuda
        self.use_elmo = use_elmo

        self.word_emb = nn.Embedding(self.vocab_size,
                                     word_embed_size,
                                     padding_idx=0)
        if not inference:
            self.word_emb.weight.data.copy_(vocab.vectors)
        if self.use_pos:
            self.pos_emb = nn.Embedding(self.pos_size,
                                        pos_embed_size,
                                        padding_idx=0)
        if self.use_elmo:
            self.elmo_size = 1024  # magic number for the hidden size of elmo representation
            self.elmo_layers = 2
            self.bilstm_input_size += self.elmo_size
            self.bilstm_input_size -= self.word_embed_size
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self.elmo = Elmo(options_file, weight_file, self.elmo_layers, dropout=0)
            self.weights = nn.Parameter(torch.zeros(self.elmo_layers, self.elmo_size))
            self.gamma = nn.Parameter(torch.ones(1))

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

    def forward(self, input_tokens, raw_tokens, input_pos, input_head, phase, compute_loss=True, log_prob=False):
        loss = 0.0
        batch_size, seq_len = input_tokens.size()
        self.init_hidden(batch_size, use_cuda=self.use_cuda)

        if self.use_elmo:
            character_ids = batch_to_ids(raw_tokens).cuda()
            x_i = self.elmo(character_ids)['elmo_representations']
            # weighting for each layer (equation. 1 in the original paper)
            x_i = torch.stack((x_i[0], x_i[1]), dim=2)
            s_task = nn.functional.softmax(self.weights, dim=0)
            s_task = s_task.expand(batch_size, seq_len, self.elmo_layers, self.elmo_size)
            x_i = x_i * s_task
            x_i = self.gamma * x_i.sum(2)
        else:
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
            if compute_loss:
                loss += self.compute_loss(p_head.squeeze(), input_head[:, i])
            if log_prob:
                parent_prob_table.append(p_head)
            else:
                parent_prob_table.append(torch.exp(p_head))

        parent_prob_table = torch.cat((parent_prob_table), dim=2).data.transpose(1, 2)
        if self.use_cuda:
            parent_prob_table = parent_prob_table.cpu()
        _, topi = parent_prob_table.topk(k=1, dim=2)
        preds = topi.squeeze()

        return loss, preds, parent_prob_table

    def compute_loss(self, prob, gold):
        return self.criterion(prob, gold)

    def attention(self, source, target, mask=None):
        function_g = \
            self.v_a_inv(torch.tanh(self.u_a(source) + self.w_a(target)))
        if mask is not None:
            function_g.masked_fill_(mask, -1e4)
        return nn.functional.log_softmax(function_g, dim=1)

    def init_hidden(self, batch_size, use_cuda):
        zeros = torch.zeros(4, batch_size, self.hidden_size, requires_grad=True)
        if use_cuda:
            self.h_n = zeros.cuda()
            self.c_n = zeros.cuda()
        else:
            self.h_n = zeros
            self.c_n = zeros
        return self.h_n, self.c_n
