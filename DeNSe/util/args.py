import argparse


def parse_args():

    gpu_id = -1
    parser = argparse.ArgumentParser(prog='DeNSe')
    parser.add_argument('--gpu-id',
                        type=int,
                        metavar='GPU_ID',
                        default=gpu_id)
    parser.add_argument('--config',
                        type=str,
                        metavar='FILENAME',
                        default='config.toml',
                        help='specify config file (default: `config.toml`)')
    return parser.parse_args()
