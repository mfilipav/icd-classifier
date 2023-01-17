import argparse
from pecos.xmc.xlinear.model import XLinearModel
from pecos.xmc import Indexer, LabelEmbeddingFactory


def construct_X_Y(notefile, Y, w2ind, c2ind):
    pass


def write_bows(out_name, X, hadm_ids, y, ind2c):
    pass


def read_bows(bow_fname, c2ind):
    pass


def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file", type=str, help="path to the training file")
    parser.add_argument(
        "--test_file", type=str,
        help="path to a dev/test file")
    parser.add_argument(
        "--vocab", type=str,
        help="path to a file holding vocab word list for discretizing words")
    parser.add_argument(
        "--model", type=str, default='log_reg',
        help="model name is log reg, just for record keeping")
    parser.add_argument(
        "--number_labels", type=str,
        help="size of label space, 'full' or '50'")
    parser.add_argument("--ngram", type=int, default=0, help="size if ngrams")
    parser.add_argument(
        "--c", type=float, default=1.0, help="log reg parameter C")
    parser.add_argument(
        "--max_iter", type=int, default=20, help="log reg max iterations")

    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)
