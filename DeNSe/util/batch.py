from torchtext import data
from torchtext.vocab import GloVe
from DeNSe.util.const import Phase


def create_dataset(data: dict, batch_size: int, device: int):

    train = Dataset(data[Phase.TRAIN]['sentence'],
                    data[Phase.TRAIN]['pos'],
                    data[Phase.TRAIN]['head'],
                    vocab=None,
                    posset=None,
                    batch_size=batch_size,
                    device=device,
                    phase=Phase.TRAIN)

    dev = Dataset(data[Phase.DEV]['sentence'],
                  data[Phase.DEV]['pos'],
                  data[Phase.DEV]['head'],
                  vocab=train.vocab,
                  posset=train.posset,
                  batch_size=batch_size,
                  device=device,
                  phase=Phase.DEV)

    test = Dataset(data[Phase.TEST]['sentence'],
                   data[Phase.TEST]['pos'],
                   data[Phase.TEST]['head'],
                   vocab=train.vocab,
                   posset=train.posset,
                   batch_size=batch_size,
                   device=device,
                   phase=Phase.TEST)
    return train, dev, test


class Dataset:
    def __init__(self,
                 sentences: list,
                 pos_list: list,
                 heads_list: list,
                 vocab: list,
                 posset: list,
                 batch_size: int,
                 device: int,
                 phase: Phase):
        assert len(sentences) == len(pos_list), \
            'the number of sentences and the number of POS/head sequences \
             should be the same length'

        self.sentences = sentences
        self.pos_list = pos_list
        # we use pseudo a heads list when the time of inference
        self.heads_list = heads_list if heads_list else [[0]] * len(sentences)
        self.sentence_id = [[i] for i in range(len(sentences))]
        self.device = device

        self.token_field = data.Field(use_vocab=True, init_token='<ROOT>', batch_first=True)
        self.pos_field = data.Field(use_vocab=True, init_token='<ROOT>', batch_first=True)
        self.head_field = data.Field(use_vocab=False, init_token=-1, pad_token=-1, batch_first=True)
        self.sentence_id_field = data.Field(use_vocab=False, batch_first=True)
        self.dataset = self._create_dataset()

        if vocab is None:
            self.token_field.build_vocab(self.sentences, vectors=GloVe(name='6B', dim=300))
            self.vocab = self.token_field.vocab
        else:
            self.token_field.vocab = vocab
            self.vocab = vocab

        if posset is None:
            self.pos_field.build_vocab(self.pos_list, min_freq=0)
            self.posset = self.pos_field.vocab
        else:
            self.pos_field.vocab = posset
            self.posset = posset

        self._set_batch_iter(batch_size, phase)

    def get_raw_sentence(self, sentences):
        return [[self.vocab.itos[idx] for idx in sentence]
                for sentence in sentences]

    def get_raw_pos(self, poss):
        return [[self.posset.itos[idx] for idx in pos]
                for pos in poss]

    def _create_dataset(self):
        _fields = [('token', self.token_field),
                   ('pos', self.pos_field),
                   ('head', self.head_field),
                   ('sentence_id', self.sentence_id_field)]
        return data.Dataset(self._get_examples(_fields), _fields)

    def _get_examples(self, fields: list):
        ex = []
        for sentence, pos, head, sentence_id in zip(self.sentences, self.pos_list, self.heads_list, self.sentence_id):
            ex.append(data.Example.fromlist([sentence, pos, head, sentence_id], fields))
        return ex

    def _set_batch_iter(self, batch_size: int, phase: Phase):

        def sort(data: data.Dataset) -> int:
            return len(getattr(data, 'token'))

        train = True if phase == Phase.TRAIN else False

        self.batch_iter = data.BucketIterator(dataset=self.dataset,
                                              batch_size=batch_size,
                                              sort_key=sort,
                                              train=train,
                                              repeat=False,
                                              device=self.device)
