import torch
from DeNSe.util.const import Phase


def train(dataset,
          parser,
          optimizer,
          batch_size,
          n_epoch,
          phase,
          use_cuda,
          compute_loss=True):

    total_loss = 0.0
    tokens = []
    poss = []
    golds = []
    preds = []
    prob_tables = []
    if phase == Phase.TRAIN:
        parser.train()
    else:
        parser.eval()

    for batch in dataset.batch_iter:
        sentence = getattr(batch, 'token')
        pos = getattr(batch, 'pos')
        gold_head = getattr(batch, 'head')
        _, seq_len = sentence.size()

        loss, pred, parent_prob_table = parser(sentence, pos, gold_head, phase, compute_loss=compute_loss)

        if phase == Phase.TRAIN:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parser.parameters(), max_norm=5)
            optimizer.step()

        if use_cuda:
            # exclude a ROOT token/tag from sentence
            sentence = sentence.data.cpu()[:, 1:].numpy()
            pos = pos.data.cpu()[:, 1:].numpy()
            # exclude a head for ROOT
            gold = gold_head[:, 1:].data.cpu().numpy()
        else:
            # exclude a ROOT token/tag from sentence
            sentence = sentence.data[:, 1:].numpy()
            pos = pos.data[:, 1:].numpy()
            # exclude a head for ROOT
            gold = gold_head[:, 1:].data.numpy()

        pred = pred.numpy()
        tokens.extend(dataset.get_raw_sentence(sentence))
        poss.extend(dataset.get_raw_pos(pos))
        golds.extend(gold)
        preds.extend(pred)
        prob_tables.extend(parent_prob_table)
        if compute_loss:
            total_loss += loss.data[0] / seq_len

    return total_loss, tokens, poss, golds, preds, prob_tables
