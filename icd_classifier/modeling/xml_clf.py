import argparse
import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from pecos.utils.featurization.text.vectorizers import Vectorizer
from pecos.xmc.xlinear.model import XLinearModel
from pecos.xmc import Indexer, LabelEmbeddingFactory
from pecos.utils import smat_util
from icd_classifier.data.data_utils import load_codes_and_descriptions
from icd_classifier.modeling import evaluation
from icd_classifier.settings import MODEL_DIR
import logging

# TODO consider removing data items without known labels! or replace labels
# bla bla
# Remove extra logging
def convert_to_idx(data, c2ind):
    logging.info(
        "Converting labels from code to idx, according to c2ind of length: {}".format(
            len(c2ind)))
    converted_data = []
    for i, item in enumerate(data):
        item_labels = []
        for label in item:
            idx = c2ind.get(label)
            if idx is not None:
                item_labels.append(idx)
            else:
                logging.error(
                    "error in row {}, when converting label: {}".format(
                        i, label))
        converted_data.append(tuple(item_labels))
    logging.info(
        "Done. First two data itmes before: {}, and after: {}".format(
            data[0:2], converted_data[0:2]))

    return converted_data


def get_list_of_label_tuples(df, label_name, return_idx=True, c2ind=None):
    logging.info("Getting list of labels")
    list_of_lists = []
    for joined_label in df[label_name].tolist():
        list_of_lists.append(tuple(joined_label.split(";")))
    if return_idx and c2ind:
        list_of_lists = convert_to_idx(list_of_lists, c2ind)
    logging.info(f"Done. Labels of first two items: {list_of_lists[0:2]}")
    return list_of_lists


def get_code_to_desc(desc_dict, c2ind):
    code_to_desc = {}
    for item in c2ind.items():
        code = item[0]
        value = desc_dict.get(code)
        if value is not None:
            code_to_desc[code] = desc_dict.get(code)
        else:
            logging.error(f"something wrong with {code}, c2ind item: {item}")
    return code_to_desc


def prepare_x_y_csr_matrices(
        train_file, test_file, number_labels, prepare_text_files=True):

    """
    Featurize text features into TFIDF or Sentencepiece vectors
    """
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    logging.info(f"Training DataFrame consists of {len(train_df)} instances.")
    logging.info(f"Testing DataFrame consists of {len(test_df)} instances.")
    logging.info(f"Columns in DF: {train_df.columns}")

    # 1. Encode labels, Y
    # prepare list of all labels:
    ind2c, desc_dict = load_codes_and_descriptions(train_file, number_labels)
    c2ind = {str(c): i for i, c in ind2c.items()}
    idx_list = list(ind2c.keys())
    # codes_list = list(c2ind.keys())

    # get ranked list of descriptions as in c2ind
    desc_dict = dict(desc_dict)

    # overwrite c2ind with data_utils.reformat()
    """
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 11., c2ind item: ('11.', 239)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 11.8, c2ind item: ('11.8', 245)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 12., c2ind item: ('12.', 282)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 13., c2ind item: ('13.', 306)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 14., c2ind item: ('14.', 333)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 15., c2ind item: ('15.', 380)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 16., c2ind item: ('16.', 443)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 17., c2ind item: ('17.', 480)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 17.0, c2ind item: ('17.0', 481)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 17.7, c2ind item: ('17.7', 489)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 22., c2ind item: ('22.', 826)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 23.9, c2ind item: ('23.9', 871)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 29.6, c2ind item: ('29.6', 1349)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 30.9, c2ind item: ('30.9', 1451)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 32., c2ind item: ('32.', 1635)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 34., c2ind item: ('34.', 1799)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 36.01, c2ind item: ('36.01', 2029)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 36.02, c2ind item: ('36.02', 2030)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 36.05, c2ind item: ('36.05', 2033)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 40.7, c2ind item: ('40.7', 2530)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 43., c2ind item: ('43.', 2759)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 45., c2ind item: ('45.', 2959)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 47.4, c2ind item: ('47.4', 3151)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 488.1, c2ind item: ('488.1', 3258)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 50., c2ind item: ('50.', 3305)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 51., c2ind item: ('51.', 3336)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 52., c2ind item: ('52.', 3436)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 54., c2ind item: ('54.', 3662)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 55., c2ind item: ('55.', 3698)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 61., c2ind item: ('61.', 4154)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 63., c2ind item: ('63.', 4268)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 64., c2ind item: ('64.', 4297)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 719.70, c2ind item: ('719.70', 4850)
    00:01:56 2023-02-02 xml_clf.py::get_code_to_desc() L60, ERROR: something wrong with 93., c2ind item: ('93.', 7371)
    """

    code_to_desc = get_code_to_desc(desc_dict, c2ind)
    assert len(code_to_desc) == len(c2ind)

    logging.info(f"c2ind len: {len(c2ind)}")
    logging.info(f"desc_dict len: {len(code_to_desc)}, dict: {code_to_desc[:2]}")

    path_corpus = 'data/processed/pecos_corpus_'+str(number_labels)+'.txt'
    if prepare_text_files:
        logging.info("Preparing text files")
        path_z = 'data/processed/Z.all.txt'
        path_x = 'data/processed/X.trn.txt'

        dfz = pd.DataFrame.from_dict(
            code_to_desc, orient='index',
            columns=['description'], dtype='object')
        dfz.iloc[:, 0].to_csv(
            path_or_buf=path_z, index=False, header=False)
        train_df.iloc[:, 2].to_csv(
            path_or_buf=path_x, index=False, header=False)

        # concate Z and X.trn into joint corpus
        # cat ./X.trn.txt Z.all.txt > pecos_50/full_corpus.txt
        corpus = pd.concat(
            [dfz.iloc[:, 0], train_df.iloc[:, 2]], axis=0, join='outer')
        corpus.to_csv(
            path_or_buf=path_corpus, index=False, header=False)

    # get labels for datesets into a list of tuples
    label_name = "LABELS"
    labels_tr = get_list_of_label_tuples(
        train_df, label_name, return_idx=True, c2ind=c2ind)
    labels_te = get_list_of_label_tuples(
        test_df, label_name, return_idx=True, c2ind=c2ind)

    # create multihot label encoding, CSR matrix
    logging.info("Preparing CSR matrices for Y train, test")
    label_encoder_multilabel = MultiLabelBinarizer(
        classes=idx_list, sparse_output=True)
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
                    "min_df_cnt": 0,  # try 5 or 3
                    "max_df_ratio": 0.98,
                    "truncate_length": -1,
                    "add_one_idf": False,  # try True
                    "analyzer": "word",
                },
            ],
        },
    }

    train_texts = [str(x) for x in train_df[text_feature].tolist()]
    test_texts = test_df[text_feature].tolist()
    # if no model found: train, save; else: load()

    # vectorizer = Vectorizer.train(
    #   trn_corpus=train_texts, config=vectorizer_config)
    logging.info(f"Train vectorizer from corpus: {path_corpus}")

    vectorizer = Vectorizer.train(
        trn_corpus=path_corpus,
        config=vectorizer_config)

    X_trn = vectorizer.predict(train_texts)
    X_tst = vectorizer.predict(test_texts)
    Z_all = vectorizer.predict(list(code_to_desc.values()))
    logging.info(
        f"Text feature: {text_feature}, train shape: {X_trn.shape} "
        "test shape: {X_tst.shape}, Z_all shape: {Z_all.shape}")

    return X_trn, X_tst, Y_trn, Y_tst, Z_all


def main(args):
    train_file, test_file = args.train_file, args.test_file
    number_labels, topk = args.number_labels, args.topk
    model, load_model_from = args.model, args.load_model_from

    X_trn, X_tst, Y_trn, Y_tst, Z_all = prepare_x_y_csr_matrices(
        train_file, test_file, number_labels,
        prepare_text_files=args.prepare_text_files)

    # prepare PIFA label embedding
    Z_pifa_concat = LabelEmbeddingFactory.create(
        Y_trn, X_trn, Z=Z_all, method="pifa_lf_concat")

    """
    nr_splits: int = 16
    min_codes: int = None  # type: ignore
    max_leaf_size: int = 100
    spherical: bool = True
    seed: int = 0
    kmeans_max_iter: int = 20
    threads: int = -1

    # paramters for sampling of hierarchical clustering
    do_sample: bool = False
    max_sample_rate: float = 1.0
    min_sample_rate: float = 0.1
    warmup_ratio: float = 0.4
    """
    cluster_chain = Indexer.gen(
        Z_pifa_concat, indexer_type="hierarchicalkmeans", nr_splits=16)

    # end cluster chain
    
    if load_model_from:
        logging.info(f"Loading model from: '{load_model_from}'")

        # Load model
        xlm = XLinearModel.load(load_model_from, is_predict_only=True)
    else:
        # Train new model
        logging.info("Train model")
        # C is cluster chain, if given, will speed up model training!
        xlm = XLinearModel.train(X=X_trn, Y=Y_trn, C=cluster_chain)

        # Save model
        model_dir = os.path.join(MODEL_DIR, '_'.join(
            [
                "pecos", model, train_file.split("/")[-1].split(".")[-2],
                "topk", str(topk),
                time.strftime('%Y%m%d_%H%M%S', time.localtime())
            ])
        )
        logging.info(f"Saving model to: '{model_dir}'")
        os.mkdir(model_dir)
        xlm.save(model_dir)

    # Predict
    logging.info("Predict")
    Y_pred_logits = xlm.predict(X_tst, beam_size=10, only_topk=topk)
    Y_pred = np.where(Y_pred_logits.toarray() > 0, 1, 0)

    # Evaluate
    logging.info("Y pred, its logits")
    logging.debug(
        "Y_pred_logits, shape: {}, 1st array: {}".format(
            Y_pred_logits.shape, Y_pred_logits[0].toarray()))
    logging.debug(
        "Y_pred, shape: {}, 1st array: {}".format(
            Y_pred.shape, Y_pred[0]))
    logging.debug(
        "Y_tst, shape: {}, 1st array: {}".format(
            Y_tst.shape, Y_tst[0].toarray()))

    logging.info("Metrics from Pecos")
    metrics = smat_util.Metrics.generate(tY=Y_tst, pY=Y_pred_logits, topk=topk)
    logging.info(f"\nrecall: {metrics.recall} \nprecision: {metrics.prec}")

    logging.info("Metrics from evaluation.all_metrics")

    metrics = evaluation.all_metrics(
        yhat=Y_pred, y=Y_tst.toarray(), k=topk, yhat_raw=Y_pred_logits.toarray())

    logging.info("metrics after evaluation: {}".format(metrics))
    evaluation.print_metrics(metrics)


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
        "--model", type=str, default='xr_linear')
    parser.add_argument(
        "--topk", type=int, default=8, help="k value for precision/recall@k")
    parser.add_argument(
        "--load_model_from", type=str, default=None)
    parser.add_argument(
        "--prepare_text_files", action="store_const",
        required=False, const=True)

    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)
