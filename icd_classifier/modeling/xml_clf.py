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


def convert_label_codes_to_idx(data, c2ind):
    """
    for all train/test examples, convert code labels to idx labels,
    according to c2ind dict:
    [
        ('801.35', '348.4', '805.06', '807.01', '998.30', '707.24',
         'E880.9','427.31', '414.01', '401.9', 'V58.61', 'V43.64',
         '707.00', 'E878.1', '96.71'),
        ('852.25', 'E888.9', '403.90', '585.9', '250.00', '414.00',
         'V45.81', '96.71')
    ]
    -->
    [
        (6106, 1910, 6204, 6241, 7906, 4683,
         8160, 2720, 2611, 2534, 8806, 8683,
         4663, 8140, 7575),
        (6775, 8186, 2545, 4009, 985, 2610,
         8717, 7575)
    ]
    """
    logging.info(
        "Convert labels from code to idx, according to c2ind of "
        "length: {}".format(len(c2ind)))
    converted_data = []
    for i, item in enumerate(data):
        item_labels = []
        for label in item:
            idx = c2ind.get(label)
            if idx is not None:
                item_labels.append(idx)
            # else:
            #     logging.warning(
            #         "label not found: {} in c2ind, data item: {}".format(
            #             label, i))
        converted_data.append(tuple(item_labels))
    logging.info(
        "Done. First two data items before conversion: {}, and after: {}"
        "".format(data[0:2], converted_data[0:2]))

    return converted_data


def get_label_tuples(df, label_name, return_idx=True, c2ind=None):
    logging.info(
        f"Getting list of labels from df['{label_name}'], "
        "return_idx={return_idx}")
    list_of_lists = []
    for joined_label in df[label_name].tolist():
        list_of_lists.append(tuple(joined_label.split(";")))
    if return_idx and c2ind:
        list_of_lists = convert_label_codes_to_idx(list_of_lists, c2ind)
    logging.info(f"Done. Labels of first two items: {list_of_lists[0:2]}")
    return list_of_lists


def get_code_to_desc_dict(desc_dict, c2ind):
    """
    returns dict with items {code: description}
    where 'code' is from c2ind dict, and
    'description' is from description dict
    'desc_dict', where keys are 'code'

    {
        '017.21': 'Tuberculosis of peripheral lymph nodes, bacteriological or
            histological examination not done',
        '017.22': 'Tuberculosis of peripheral lymph nodes..',
    }
    """
    code_to_desc_dict = {}
    for item in c2ind.items():
        code = item[0]
        value = desc_dict.get(code)
        if value is not None:
            code_to_desc_dict[code] = desc_dict.get(code)
        else:
            logging.error(f"something wrong with {code}, c2ind item: {item}")
    return code_to_desc_dict


def prepare_x_z_corpus_files(
        train_df, code_to_desc_dict, path_x, path_z, path_corpus):
    dfz = pd.DataFrame.from_dict(
        code_to_desc_dict, orient='index',
        columns=['description'], dtype='object')
    dfz.iloc[:, 0].to_csv(
        path_or_buf=path_z, index=False, header=False)
    logging.info(
        f"Prepared label description file {path_z}, line number "
        "corresponds to code in c2ind")
    train_df.iloc[:, 2].to_csv(
        path_or_buf=path_x, index=False, header=False)
    logging.info(f"Prepared training text file: {path_x}")

    # concate Z and X.trn into joint corpus
    # cat ./X.trn.txt Z.all.txt > pecos_50/full_corpus.txt
    corpus = pd.concat(
        [dfz.iloc[:, 0], train_df.iloc[:, 2]], axis=0, join='outer')
    corpus.to_csv(
        path_or_buf=path_corpus, index=False, header=False)
    logging.info(
        f"Prepared a joint training text and label \
        description corpus: {path_corpus}")


def encode_y_labels(train_df, test_df, label_name, c2ind, ind_list):
    # prepare list of all labels:
    # get labels for datesets into a list of tuples
    labels_tr = get_label_tuples(
        train_df, label_name, return_idx=True, c2ind=c2ind)
    labels_te = get_label_tuples(
        test_df, label_name, return_idx=True, c2ind=c2ind)

    # create multihot label encoding, CSR matrix
    logging.info("Preparing CSR matrices for Y train, test")
    label_encoder_multilabel = MultiLabelBinarizer(
        classes=ind_list, sparse_output=True)
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
    return Y_trn, Y_tst


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

    ind2c, desc_dict = load_codes_and_descriptions(train_file, number_labels)
    c2ind = {str(c): i for i, c in ind2c.items()}
    ind_list = list(ind2c.keys())
    logging.info(f"c2ind len: {len(c2ind)}")

    # 0. Prepare X.trn, corpus, ICD-9 codes' descriptions
    # get a dict of descriptions as in c2ind, same order as in c2ind
    code_to_desc_dict = get_code_to_desc_dict(dict(desc_dict), c2ind)
    logging.info(f"code_to_desc_dict len: {len(code_to_desc_dict)}")
    assert len(code_to_desc_dict) == len(c2ind)

    path_corpus = 'data/processed/pecos_corpus_'+str(number_labels)+'.txt'
    path_z = 'data/processed/Z.all.txt'
    path_x = 'data/processed/X.trn.txt'
    if prepare_text_files:
        prepare_x_z_corpus_files(
            train_df, code_to_desc_dict, path_x, path_z, path_corpus)

    # 1. Encode labels, Y
    label_name = "LABELS"
    Y_trn, Y_tst = encode_y_labels(
        train_df, test_df, label_name, c2ind, ind_list)

    # 2. Encode text features, X
    # TODO: parameterize TFIDF, NGRAM range
    # we adopt n-gram TFIDF features containing word unigrams,
    # word bigrams, and character trigrams.
    # Note that each of the n-gram feature can have different hyper-parameters,
    # such as max_feature and max_df, based on corpus size!!    
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
    logging.info(f"Train vectorizer from corpus: {path_corpus}")
    vectorizer = Vectorizer.train(
        trn_corpus=path_corpus,
        config=vectorizer_config)

    # 4. Fit vectorizer to X train and test, and Z
    X_trn = vectorizer.predict(train_texts)
    logging.info(
        f"The train file consists of {X_trn.shape[0]} instances \
        with {X_trn.shape[1]}-dimensional features \
        in a {X_trn.getformat()} matrix.")
    X_tst = vectorizer.predict(test_texts)
    Z_all = vectorizer.predict(list(code_to_desc_dict.values()))
    logging.info(
        f"Text feature: {text_feature}, train shape: {X_trn.shape}, \
        test shape: {X_tst.shape}, Z_all shape: {Z_all.shape}")

    return X_trn, X_tst, Y_trn, Y_tst, Z_all


def main(args):
    train_file, test_file = args.train_file, args.test_file
    number_labels, topk = args.number_labels, args.topk
    model, load_model_from, hlt = args.model, args.load_model_from, args.hlt

    X_trn, X_tst, Y_trn, Y_tst, Z_all = prepare_x_y_csr_matrices(
        train_file, test_file, number_labels,
        prepare_text_files=args.prepare_text_files)

    if hlt:
        logging.info("Prepare PIFA hierarchical label tree (HLT)")
        Z_pifa_concat = LabelEmbeddingFactory.create(
            Y_trn, X_trn, Z=Z_all, method="pifa_lf_concat")

        logging.info("Generate cluster chain with PIFA, hierarchical k means")
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
            Z_pifa_concat, indexer_type="hierarchicalkmeans",
            max_leaf_size=128, nr_splits=128, verbose=1)
        logging.info(
            f"{len(cluster_chain)} layers in the trained hierarchical label \
            tree with C[d] as:")
        for d, C in enumerate(cluster_chain):
            logging.info(
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
            X=X_trn, Y=Y_trn, C=cluster_chain,
            negative_sampling_scheme="tfn")

        # Save model
        hlt_flag = 'hlt' if hlt else ''
        model_dir = os.path.join(MODEL_DIR, '_'.join(
            [
                "pecos", model, train_file.split("/")[-1].split(".")[-2],
                "topk", str(topk), hlt_flag,
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
    logging.debug(
        "Y_pred_logits, shape: {}, 1st array: {}".format(
            Y_pred_logits.shape, Y_pred_logits[0].toarray()))
    logging.debug(
        "Y_pred, shape: {}, 1st array: {}".format(
            Y_pred.shape, Y_pred[0]))
    logging.debug(
        "Y_tst, shape: {}, 1st array: {}".format(
            Y_tst.shape, Y_tst[0].toarray()))

    logging.info("Results from Pecos:")
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
        "--prepare_text_files", action="store_const", required=False,
        const=True)
    parser.add_argument(
        "--hlt", action="store_const", required=False, const=True,
        help="obtain hierarchical label tree with PIFA")

    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)
