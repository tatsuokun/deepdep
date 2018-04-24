def load_conllx(file_name: str):

    conllx = {}
    sentence = []
    pos_seq = []
    heads_seq = []
    token = []
    pos = []
    head = []

    with open(file_name, mode='r') as f:
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
    conllx['sentence'] = sentence
    conllx['pos'] = pos_seq
    conllx['head'] = heads_seq

    return conllx


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
