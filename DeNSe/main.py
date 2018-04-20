import os
import torch
from torch import optim

from DeNSe.core.dependency_parser import DependencyParser
from DeNSe.core.train import train
from DeNSe.util.args import parse_args
from DeNSe.util.batch import create_dataset
from DeNSe.util.config import Config, output_model_config, save_vocab
from DeNSe.util.const import Phase
from DeNSe.util.conllx import load_ptb_conllx, output_conllx_format
from DeNSe.util.model_func import save_model


def main():

    dataset_dir = 'DeNSe/data'
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    model_dir = 'DeNSe/models'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    args = parse_args()
    config = Config(args.config)

    gpu_id = args.gpu_id
    use_cuda = torch.cuda.is_available() and gpu_id > -1
    use_pos = True  # TODO; move to args

    ptb = load_ptb_conllx()
    data_train, data_dev, data_test = \
        create_dataset(ptb, batch_size=config.batch_size, device=gpu_id)
    parser = DependencyParser(vocab=data_train.vocab,
                              n_pos=data_train.n_pos,
                              word_embed_size=config.word_embed_size,
                              pos_embed_size=config.pos_embed_size,
                              hidden_size=config.hidden_size,
                              use_pos=use_pos,
                              use_cuda=use_cuda)
    if use_cuda:
        torch.cuda.set_device(gpu_id)
        parser = parser.cuda()

    optimizer = optim.Adam(parser.parameters(), lr=config.learning_rate)

    for epoch in range(config.n_epochs):
        train_loss, _, _, _, _, _ = train(data_train,
                                          parser,
                                          optimizer,
                                          config.batch_size,
                                          epoch,
                                          Phase.TRAIN,
                                          use_cuda)

        dev_loss, sentences, poss, golds, preds, _ = train(data_dev,
                                                           parser,
                                                           optimizer,
                                                           config.batch_size,
                                                           epoch,
                                                           Phase.DEV,
                                                           use_cuda)
        logger = '\t'.join(['epoch {}'.format(epoch+1),
                            'TRAIN Loss: {:.2f}'.format(train_loss),
                            'DEV Loss: {:.2f}'.format(dev_loss)])
        print(logger)

    output_conllx_format(sentences, poss, golds, 'DeNSe/data/dev_gold')
    output_conllx_format(sentences, poss, preds, 'DeNSe/data/dev_pred')

    _, sentences, poss, golds, preds, _ = train(data_test,
                                                parser,
                                                optimizer,
                                                config.batch_size,
                                                epoch,
                                                Phase.TEST,
                                                use_cuda)

    output_conllx_format(sentences, poss, golds, 'DeNSe/data/test_gold')
    output_conllx_format(sentences, poss, preds, 'DeNSe/data/test_pred')

    model_config_name = os.path.join(model_dir, 'model_config.toml')
    output_model_config(batch_size=config.batch_size,
                        n_pos=data_train.n_pos,
                        word_embed_size=config.word_embed_size,
                        pos_embed_size=config.pos_embed_size,
                        hidden_size=config.hidden_size,
                        use_pos=use_pos,
                        learning_rate=config.learning_rate,
                        save_to=model_config_name)

    model_params_name = os.path.join(model_dir, 'model_params.pth')
    save_model(parser, model_params_name)
    optim_params_name = os.path.join(model_dir, 'optim_params.pth')
    save_model(optimizer, optim_params_name)
    vocab_file_name = os.path.join(model_dir, 'vocab.pkl')
    save_vocab(data_train.vocab, vocab_file_name)
