import toml
import pickle


class Config:

    def __init__(self, filename: str):

        self.filename = filename
        config = toml.load(self.filename)

        trainer = config.get('config', {})
        self.n_epochs = int(trainer.get('n_epochs', 15))
        self.batch_size = int(trainer.get('batch_size', 16))
        self.word_embed_size = int(trainer.get('embed_size', 300))
        self.pos_embed_size = int(trainer.get('time_embed_size', 30))
        self.hidden_size = int(trainer.get('hidden_size', 300))
        self.learning_rate = float(trainer.get('learning_rate', 1e-3))

        dataset = config.get('dataset', {})
        self.train_file = dataset.get('train_file', '')
        self.dev_file = dataset.get('dev_file', '')
        self.test_file = dataset.get('test_file', '')
        assert self.train_file, 'make sure your train/dev/test files in ' + self.filename


class ModelConfig:

    def __init__(self, filename: str):

        self.filename = filename
        config = toml.load(self.filename)

        model = config.get('model', {})
        self.batch_size = int(model.get('batch_size', 16))
        self.word_embed_size = int(model.get('embed_size', 300))
        self.pos_embed_size = int(model.get('time_embed_size', 30))
        self.hidden_size = int(model.get('hidden_size', 300))
        self.use_pos = bool(model.get('use_pos', False))
        self.use_elmo = bool(model.get('use_elmo', False))

        optim = config.get('optim', {})
        self.learning_rate = float(optim.get('learning_rate', 1e-3))

    def set_vocab(self, vocab):
        self.vocab = vocab

    def set_pos(self, pos):
        self.pos = pos

    def return_parser_param(self, use_cuda: bool, inference: bool):
        return [self.vocab,
                self.pos,
                self.word_embed_size,
                self.pos_embed_size,
                self.hidden_size,
                self.use_pos,
                self.use_elmo,
                use_cuda,
                inference]

    def return_learning_rate(self):
        return self.learning_rate


def output_model_config(batch_size: int,
                        word_embed_size: int,
                        pos_embed_size: int,
                        hidden_size: int,
                        use_pos: bool,
                        use_elmo: bool,
                        learning_rate: float,
                        save_to: str):
    config = dict()
    config['model'] = {'batch_size': batch_size,
                       'word_embed_size': word_embed_size,
                       'pos_embed_size': pos_embed_size,
                       'hidden_size': hidden_size,
                       'use_pos': use_pos,
                       'use_elmo': use_elmo}
    config['optim'] = {'learning_rate': learning_rate}
    toml_string = toml.dumps(config)
    with open(save_to, mode='w') as w:
        w.writelines(toml_string)


def save_vocab(vocab: int, save_to: str):
    with open(save_to, mode='wb') as w:
        pickle.dump(vocab, w)


def load_vocab(vocab_file: str):
    with open(vocab_file, mode='rb') as f:
        return pickle.load(f)
