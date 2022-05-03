from trainer import train
from tester import test
import argparse


def args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('config', type=str)
    argument_parser.add_argument('--test', type=bool)
    argument_parser.add_argument('--resume', type=bool)
    argument_parser.add_argument('--version', type=int)
    argument_parser.add_argument('--batch', type=int)
    argument_parser.add_argument('--checkpath', type=str)

    parsed_args = argument_parser.parse_args()

    if parsed_args.test is not None:
        test(parsed_args.config, parsed_args.checkpath, parsed_args.batch, parsed_args.version)
    else:
        train(parsed_args.config, parsed_args.resume is not None, parsed_args.version, parsed_args.batch,
              parsed_args.checkpath)


if __name__ == '__main__':
    args()
