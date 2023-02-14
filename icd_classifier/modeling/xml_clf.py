import argparse
import sys
import os
import time
import logging
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pecos.utils.featurization.text.vectorizers import Vectorizer
from pecos.xmc.xlinear.model import XLinearModel
from pecos.xmc import Indexer, LabelEmbeddingFactory
from pecos.utils import smat_util
from icd_classifier.data.data_utils import load_codes_and_descriptions
from icd_classifier.modeling import evaluation
from icd_classifier.modeling import tools
from icd_classifier.settings import MODEL_DIR, DATA_DIR
from icd_classifier.modeling.tools import (
    get_code_to_desc_dict, prepare_x_z_corpus_files, encode_y_labels)


def prepare_x_y_csr_matrices(
        train_file, test_file, number_labels,
        prepare_text_files, label_feat_path):

    """
    Featurize text features into TFIDF or Sentencepiece vectors
    """
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    logging.info(f"Training DataFrame consists of {len(train_df)} instances.")
    logging.info(f"Testing DataFrame consists of {len(test_df)} instances.")
    logging.info(f"Columns in DF: {train_df.columns}")

    ind2c, desc_dict = load_codes_and_descriptions(train_file, number_labels)
    c2ind = {str(c): i for i, c in ind2c.items()}
    ind_list = list(ind2c.keys())
    logging.info(f"c2ind len: {len(c2ind)}")

    # 0. Prepare X.trn, corpus, ICD-9 codes' descriptions
    # get a dict of descriptions as in c2ind, same order as in c2ind
    code_to_desc_dict = get_code_to_desc_dict(dict(desc_dict), c2ind)
    logging.info(f"code_to_desc_dict len: {len(code_to_desc_dict)}")
    assert len(code_to_desc_dict) == len(c2ind)

    path_corpus = DATA_DIR+'/pecos_corpus.'+str(number_labels)+'.txt'
    path_z = DATA_DIR+'/Z.all.'+str(number_labels)+'.txt'
    path_x_trn = DATA_DIR+'/X.trn.'+str(number_labels)+'.txt'
    path_x_tst = DATA_DIR+'/X.tst.'+str(number_labels)+'.txt'
    path_code_to_desc_dict = \
        DATA_DIR+'/code_to_desc_dict.'+str(number_labels)+'.json'
    path_c2ind_dict = DATA_DIR+'/c2ind_dict.'+str(number_labels)+'.json'

    if prepare_text_files:
        prepare_x_z_corpus_files(
            train_df, test_df, code_to_desc_dict, path_x_trn, path_x_tst,
            path_z, path_corpus)
        with open(path_code_to_desc_dict, "w") as outfile:
            json.dump(code_to_desc_dict, outfile)
        with open(path_c2ind_dict, "w") as outfile:
            json.dump(c2ind, outfile)

    # 1. Encode labels, Y
    label_name = "LABELS"
    Y_trn, Y_tst = encode_y_labels(
        train_df, test_df, label_name, c2ind,
        ind_list, prepare_text_files, number_labels)

    # 2. Encode text features, X
    # TODO: parameterize TFIDF, NGRAM range
    # see pecos/utils/featurization/text/vectorizers.py::train()
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
                    "smooth_idf": True,
                    "add_one_idf": False,
                    "analyzer": "word",
                },
                # {
                #     "ngram_range": [2, 2],
                #     "max_df_ratio": 0.98,
                #     "analyzer": "word",
                # },
                # {
                #     "ngram_range": [3, 3],
                #     "max_df_ratio": 0.98,
                #     "analyzer": "char_wb",
                # },
            ],
        },
    }

    train_texts = [str(x) for x in train_df[text_feature].tolist()]
    test_texts = test_df[text_feature].tolist()

    # 3. Train vectorizer
    # vectorizer = Vectorizer.train(
    #   trn_corpus=train_texts, config=vectorizer_config)
    logging.info(
        "Train {} vectorizer from corpus: {}".format(
            vectorizer_config.get("type"), path_corpus))
    vectorizer = Vectorizer.train(
        trn_corpus=path_corpus,
        config=vectorizer_config)

    # 4. Fit vectorizer to X train and test, and Z
    X_trn = vectorizer.predict(train_texts)
    logging.info(
        f"The train file consists of {X_trn.shape[0]} instances "
        f"with {X_trn.shape[1]}-dimensional features "
        f"in a {X_trn.getformat()} matrix.")
    X_tst = vectorizer.predict(test_texts)

    if label_feat_path:
        logging.info(
            f"Load Z label embeddings from:{label_feat_path}")
        Z_all = XLinearModel.load_feature_matrix(
            label_feat_path)
    else:
        # Prepare TFIDF label embeddings
        Z_all = vectorizer.predict(list(code_to_desc_dict.values()))
    logging.info(
        f"Text feature: {text_feature}, X_trn shape: {X_trn.shape}, "
        f"X_trn type: {type(X_trn)}, "
        f"X_tst shape: {X_tst.shape}, Z_all shape: {Z_all.shape}, "
        f"Z_all type: {type(Z_all)}")

    return X_trn, X_tst, Y_trn, Y_tst, Z_all


def main(args):
    train_file, test_file = args.train_file, args.test_file
    number_labels, topk = args.number_labels, args.topk
    b_partitions = args.b_partitions
    model, load_model_from, hlt = args.model, args.load_model_from, args.hlt
    label_feat_path = args.label_feat_path
    prepare_text_files = args.prepare_text_files

    X_trn, X_tst, Y_trn, Y_tst, Z_all = prepare_x_y_csr_matrices(
        train_file, test_file, number_labels,
        prepare_text_files,
        label_feat_path)

    if hlt:
        logging.info(
            "Prepare hierarchical label tree (HLT), label features Z and "
            "cluster chain C")
        if label_feat_path:
            logging.info("Construct label features Z from embedding: "
                         "{}".format(label_feat_path))
            Z = Z_all
        else:
            logging.info("Construct label features Z by applying PIFA "
                         "clustering")
            Z = LabelEmbeddingFactory.create(
                Y_trn, X_trn, Z=Z_all, method="pifa_lf_concat")

        logging.info("Recursively generate label cluster chain, with cluster"
                     f"size B={b_partitions}")
        cluster_chain = Indexer.gen(
            feat_mat=Z, indexer_type="hierarchicalkmeans",
            max_leaf_size=100, nr_splits=b_partitions,
            spherical=True, do_sample=False, verbose=3)
        logging.debug(
            f"{len(cluster_chain)} layers in the trained hierarchical label "
            "tree with C[d] as:")
        for d, C in enumerate(cluster_chain):
            logging.debug(
                f"cluster_chain[{d}] is a {C.getformat()} matrix of \
                shape {C.shape}")
    else:
        cluster_chain = None

    if load_model_from:
        logging.info(f"Loading model from: '{load_model_from}'")
        xlm = XLinearModel.load(load_model_from, is_predict_only=True)
    else:
        logging.info("Train a new XLinearModel")
        # C is cluster chain, if given, will speed up model training!
        xlm = XLinearModel.train(
            X=X_trn, Y=Y_trn, c=cluster_chain,
            negative_sampling_scheme="tfn", verbose=3)

        for d, m in enumerate(xlm.model.model_chain):
            logging.debug(
                f"model_chain[{d}].W is a {m.W.getformat()} "
                f"matrix of shape {m.W.shape}")

        # Save model
        hlt_flag = 'hlt' if hlt else ''
        model_dir = os.path.join(MODEL_DIR, '_'.join(
            [
                "xmc", model, train_file.split("/")[-1].split(".")[-2],
                "topk", str(topk), hlt_flag,
                "b_partitions", str(b_partitions),
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
    Y_trn_pred_logits = xlm.predict(X_trn, beam_size=10, only_topk=topk)
    Y_trn_pred = np.where(Y_trn_pred_logits.toarray() > 0, 1, 0)

    # Evaluate
    logging.debug(
        "Y_pred_logits, shape: {}, 1st array: {}".format(
            Y_pred_logits.shape, Y_pred_logits[0].toarray()))
    logging.debug(
        "Y_pred, shape: {}, 1st array nnz: {}".format(
            Y_pred.shape, np.count_nonzero(Y_pred[0])))
    logging.debug(
        "Y_tst, shape: {}, 1st array: {}".format(
            Y_tst.shape, Y_tst[0].toarray()))

    logging.info("Testing Results from Pecos:")
    metrics = smat_util.Metrics.generate(tY=Y_tst, pY=Y_pred_logits, topk=topk)
    logging.info(f"\nrecall: {metrics.recall} \nprecision: {metrics.prec}")

    logging.info("Metrics from evaluation.all_metrics")
    metrics = evaluation.all_metrics(
        yhat=Y_pred, y=Y_tst.toarray(),
        k=topk, yhat_raw=Y_pred_logits.toarray())
    logging.info("metrics after TEST/DEV evaluation: {}".format(metrics))
    evaluation.print_metrics(metrics)

    metrics_tr = evaluation.all_metrics(
        yhat=Y_trn_pred, y=Y_trn.toarray(),
        k=topk, yhat_raw=Y_trn_pred_logits.toarray())
    logging.info("metrics after TRAIN evaluation: {}".format(metrics))
    evaluation.print_metrics(metrics_tr)

    # Save metrics to model dir
    logging.info("Preparing and saving metrics")
    metrics_hist = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])
    for name in metrics.keys():
        metrics_hist[name].append(metrics[name])
    for name in metrics_tr.keys():
        metrics_hist_tr[name].append(metrics_tr[name])
    metrics_hist_all = (metrics_hist, metrics_hist, metrics_hist_tr)
    tools.save_metrics(metrics_hist_all, model_dir)


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
        "--b_partitions", type=int, default=16,
        help="B value for B-ary (recursive) partitioning of label set")
    parser.add_argument(
        "--load_model_from", type=str, default=None)
    parser.add_argument(
        "--label_feat_path", type=str, default=None)
    parser.add_argument(
        "--prepare_text_files", action="store_const", required=False,
        const=True)
    parser.add_argument(
        "--hlt", action="store_const", required=False, const=True,
        help="obtain hierarchical label tree with PIFA")

    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)
