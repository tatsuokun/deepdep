from DeNSe.util.const import Phase


def load_ptb_conllx(train_file_name: str,
                    dev_file_name: str,
                    test_file_name: str):
    ptb = {}

    for phase in [Phase.TRAIN, Phase.DEV, Phase.TEST]:
        if phase == Phase.TRAIN:
            file_path = train_file_name
        elif phase == Phase.DEV:
            file_path = dev_file_name
        else:
            file_path = test_file_name

        ptb[phase] = {}
        sentence = []
        pos_seq = []
        heads_seq = []
        token = []
        pos = []
        head = []

        with open(file_path, mode='r') as f:
            for line in f:
                contents = line.split('\t')
                if contents[0].isdigit():
                    token.append(contents[1])
                    pos.append(contents[3])
                    head.append(int(contents[-1].strip()))
                else:
                    # end of sentence
                    sentence.append(token)
                    pos_seq.append(pos)
                    heads_seq.append(head)
                    token = []
                    pos = []
                    head = []
                    continue
        ptb[phase]['sentence'] = sentence
        ptb[phase]['pos'] = pos_seq
        ptb[phase]['head'] = heads_seq
    return ptb


def output_conllx_format(sentences, poss, heads, output_file_name, pad_token='<pad>'):
    assert (len(sentences) == len(poss) == len(heads))

    with open(output_file_name, 'w') as w:
        for sentence, _pos, _head in zip(sentences, poss, heads):
            assert (len(sentence) == len(_pos) == len(_head))

            for token_id, (token, pos, head) in enumerate(zip(sentence, _pos, _head)):
                if token == pad_token:
                    break
                w.write('\t'.join([str(token_id+1), token, '_', pos, pos, '_', str(head)]) + '\n')

            w.write('\n')
