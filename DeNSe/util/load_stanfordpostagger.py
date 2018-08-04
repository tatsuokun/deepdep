def load_stanford_postagger_format(file_name: str):
    data = {}
    sentence = []
    pos = []

    with open(file_name, mode='r') as f:
        for line in f:
            try:
                tokens, poss = zip(*[content.split('_') for content in line.split()])
            except ValueError:
                continue
            sentence.append([token.lower() for token in tokens])
            pos.append(list(poss))
    data['sentence'] = sentence
    data['pos'] = pos
    return data
