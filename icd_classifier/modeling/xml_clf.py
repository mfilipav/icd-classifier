import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from pecos.utils.featurization.text.vectorizers import Vectorizer
from pecos.xmc.xlinear.model import XLinearModel
from pecos.utils import smat_util
from icd_classifier.data.data_utils import load_codes_and_descriptions
import logging


def get_list_of_label_tuples(df, label_name):
    list_of_lists = []
    for joined_label in df[label_name].tolist():
        list_of_lists.append(tuple(joined_label.split(";")))
    return list_of_lists


def prepare_x_y_csr_matrices(train_file, test_file, number_labels):

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    logging.info(f"Training DataFrame consists of {len(train_df)} instances.")
    logging.info(f"Testing DataFrame consists of {len(test_df)} instances.")
    logging.info(f"Columns in DF: {train_df.columns}")

    # 1. Encode labels, Y
    # prepare list of all labels:
    ind2c, desc_dict = load_codes_and_descriptions(train_file, number_labels)
    c2ind = {c: i for i, c in ind2c.items()}
    codes_list = list(c2ind.keys())

    # get labels for datesets into a list of tuples
    label_name = "LABELS"
    labels_tr = get_list_of_label_tuples(train_df, label_name)
    labels_te = get_list_of_label_tuples(test_df, label_name)

    # create multihot label encoding, CSR matrix
    label_encoder_multilabel = MultiLabelBinarizer(
        classes=codes_list, sparse_output=True)
    Y_trn = label_encoder_multilabel.fit_transform(labels_tr)
    Y_tst = label_encoder_multilabel.fit_transform(labels_te)

    # cast as correct dtype
    Y_trn = Y_trn.astype(dtype=np.float32, copy=False)
    Y_tst = Y_tst.astype(dtype=np.float32, copy=False)

    logging.info(
        f"Y_trn is a {Y_trn.getformat()} matrix with a shape {Y_trn.shape} "
        "and {Y_trn.nnz} non-zero values.")
    logging.info(
        f"Y_tst is a {Y_tst.getformat()} matrix with a shape {Y_tst.shape} "
        "and {Y_tst.nnz} non-zero values.")

    # 2. Encode text features, X
    # TODO: parameterize TFIDF, NGRAM range
    text_feature = "TEXT"
    vectorizer_config = {
        "type": "tfidf",
        "kwargs": {
            "base_vect_configs": [
                {
                    "ngram_range": [1, 1],
                    "max_df_ratio": 0.98,
                    "analyzer": "word",
                },
            ],
        },
    }

    train_texts = [str(x) for x in train_df[text_feature].tolist()]
    test_texts = test_df[text_feature].tolist()
    vectorizer = Vectorizer.train(train_texts, config=vectorizer_config)
    X_trn = vectorizer.predict(train_texts)
    X_tst = vectorizer.predict(test_texts)
    logging.info(
        f"Text feature: {text_feature}, train shape: {X_trn.shape} "
        "and test shape: {X_tst.shape}.")

    return X_trn, X_tst, Y_trn, Y_tst


def main(args):
    train_file, test_file = args.train_file, args.test_file
    number_labels, topk = args.number_labels, args.topk

    X_trn, X_tst, Y_trn, Y_tst = prepare_x_y_csr_matrices(
        train_file, test_file, number_labels)

    # Train model
    logging.info("Train model")
    xlm = XLinearModel.train(X_trn, Y_trn)

    # Predict
    logging.info("Predict")
    Y_pred = xlm.predict(X_tst, beam_size=10, only_topk=topk)

    # Evaluate
    logging.info("Evaluate")
    metrics = smat_util.Metrics.generate(Y_tst, Y_pred, topk=topk)

    # Results
    logging.info("Metrics")
    logging.info(f"\nprecision: {metrics.prec},\nrecall: {metrics.recall}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file", type=str, help="path to the training file")
    parser.add_argument(
        "--test_file", type=str, help="path to a dev/test file")
    parser.add_argument(
        "--number_labels", type=str,
        help="size of label space, 'full' or '50'")
    parser.add_argument(
        "--topk", type=int, default=5, help="k value for precision/recall@k")

    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)
