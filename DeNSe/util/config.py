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


class ModelConfig:

    def __init__(self, filename: str):

        self.filename = filename
        config = toml.load(self.filename)

        model = config.get('model', {})
        self.batch_size = int(model.get('batch_size', 16))
        self.n_pos = int(model.get('n_pos', 16))
        self.word_embed_size = int(model.get('embed_size', 300))
        self.pos_embed_size = int(model.get('time_embed_size', 30))
        self.hidden_size = int(model.get('hidden_size', 300))
        self.use_pos = bool(model.get('use_pos', False))

        optim = config.get('optim', {})
        self.learning_rate = float(optim.get('learning_rate', 1e-3))

    def set_vocab(self, vocab):
        self.vocab = vocab

    def return_parser_param(self, use_cuda: bool):
        return [self.vocab,
                self.n_pos,
                self.word_embed_size,
                self.pos_embed_size,
                self.hidden_size,
                self.use_pos,
                use_cuda]

    def return_learning_rate(self):
        return self.learning_rate


def output_model_config(batch_size: int,
                        n_pos: int,
                        word_embed_size: int,
                        pos_embed_size: int,
                        hidden_size: int,
                        use_pos: bool,
                        learning_rate: float,
                        save_to: str):
    config = dict()
    config['model'] = {'batch_size': batch_size,
                       'n_pos': n_pos,
                       'word_embed_size': word_embed_size,
                       'pos_embed_size': pos_embed_size,
                       'hidden_size': hidden_size,
                       'use_pos': use_pos}
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
