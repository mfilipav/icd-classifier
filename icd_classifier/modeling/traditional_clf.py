"""
    Reads (or writes) BOW-formatted notes, and performs
    scikit-learn classification: logistic regression, SVC
"""
import os
import sys
import time
import argparse
from collections import defaultdict
import logging
from tqdm import tqdm
import csv
import numpy as np
import nltk
from scipy.sparse import csr_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from icd_classifier.settings import MODEL_DIR, DATA_DIR
from icd_classifier.data import data_utils
import evaluation
from icd_classifier.modeling import tools


def construct_X_Y(notefile, Y, w2ind, c2ind):
    """
        Each row in notesfile consists of text pertaining to one admission
        OneVsRestClassifier can also be used for multilabel classification.
        To use this feature, provide an indicator matrix for the target y when
        calling .fit. In other words, the target labels should be formatted
        as a 2D binary (0/1) matrix, where [i, j] == 1 indicates the presence
        of label j in sample i. This estimator uses the binary relevance method
        to perform multilabel clf, which involves training one binary
        classifier independently for each label

        INPUTS:
            notefile: path to preprocessed file containing the sub and hadm id
                notes text, labels and length
            Y: size of the output label space
            w2ind: dictionary from words to integers for discretizing
            c2ind: dictionary from labels to integers for discretizing
        OUTPUTS:
            csr_matrix where each row is a BOW
                Dimension: (# samples in dataset) x (vocab size)
            yy indicator matrix, training set, in shape [8066, 50]


        notefile example:
        SUBJECT_ID,HADM_ID,TEXT,LABELS,length
        7908, 182396, admission date discharge date date of birthtervice
            historyof the present illness this is a year old man with a ..,
        287.5;45.13;584.9, 105
    """
    # assert Y == len(c2ind)  # True, if top 50 labels
    yy = []
    hadm_ids = []
    logging.info("Start constructing crs_matrix from notes file: {}, Y: {}, "
                 "{} words in w2ind and {} codes in c2ind".format(
                    notefile, Y, len(w2ind), len(c2ind)))
    with open(notefile, 'r') as notesfile:
        reader = csv.reader(notesfile)
        next(reader)
        i = 0

        subj_inds = []
        indices = []
        data = []

        for i, row in tqdm(enumerate(reader)):
            label_set = set()
            for label in str(row[3]).split(';'):
                if label in c2ind.keys():
                    label_set.add(c2ind[label])
            subj_inds.append(len(indices))
            yy.append([1 if j in label_set else 0 for j in range(Y)])
            text = row[2]
            for word in text.split():
                if word in w2ind:
                    index = w2ind[word]
                    if index != 0:
                        # ignore padding characters
                        indices.append(index)
                        data.append(1)
                else:
                    # OOV
                    indices.append(len(w2ind))
                    data.append(1)
            i += 1
            hadm_ids.append(int(row[1]))
        subj_inds.append(len(indices))

    return csr_matrix((data, indices, subj_inds)), np.array(yy), hadm_ids


def write_bows(out_name, X, hadm_ids, y, ind2c):
    with open(out_name, 'w') as of:
        logging.info("Writing BOW CRS matrix to file: {}".format(out_name))
        w = csv.writer(of)
        w.writerow(['HADM_ID', 'BOW', 'LABELS'])
        for i in range(X.shape[0]):
            bow = X[i].toarray()[0]
            inds = bow.nonzero()[0]
            counts = bow[inds]
            bow_str = ' '.join(
                ['%d:%d' % (ind, count) for ind, count in zip(inds, counts)])
            code_str = ';'.join([ind2c[ind] for ind in y[i].nonzero()[0]])
            w.writerow([str(hadm_ids[i]), bow_str, code_str])


def read_bows(bow_fname, c2ind):
    num_labels = len(c2ind)
    data = []
    row_ind = []
    col_ind = []
    hids = []
    y = []
    logging.info("Reading BOWS from file: {}".format(bow_fname))
    with open(bow_fname, 'r') as f:
        r = csv.reader(f)
        # header
        next(r)
        for i, row in tqdm(enumerate(r)):
            if i % 10000 == 0:
                logging.debug("Read bow file row {}: {}".format(i, row))
            hid = int(row[0])
            bow_str = row[1]
            code_str = row[2]
            for pair in bow_str.split():
                split = pair.split(':')
                ind, count = split[0], split[1]
                data.append(int(count))
                row_ind.append(i)
                col_ind.append(int(ind))
            label_set = set([c2ind[c] for c in code_str.split(';')])
            y.append([1 if j in label_set else 0 for j in range(num_labels)])
            hids.append(hid)
        X = csr_matrix((data, (row_ind, col_ind)))
    return X, np.array(y), hids


def calculate_top_ngrams(inputfile, clf, c2ind, w2ind,
                         labels_with_examples, ngram):

    # Reshape the coefficients matrix back into having 0's
    # for columns of codes not in training set.
    labels_with_examples = set(labels_with_examples)
    mat = clf.coef_
    mat_full = np.zeros((8922, mat.shape[1]))
    j = 0
    for i in range(mat_full.shape[0]):
        if i in labels_with_examples:
            mat_full[i, :] = mat[j, :]
            j += 1

    # write out to csv
    top_ngrams_file = "%s/top_ngrams.csv" % DATA_DIR
    f = open(top_ngrams_file, 'wb')
    writer = csv.writer(f, delimiter=',')
    # write header
    writer.writerow([
        'SUBJECT_ID', 'HADM_ID', 'LABEL', 'INDEX', 'NGRAM', 'SCORE'])
    logging.info("Write out top ngrams to file: {}".format(
        top_ngrams_file))    
    # get text as list of strings for each record in dev set
    with open("%s" % (inputfile), 'r') as notesfile:
        reader = csv.reader(notesfile)
        next(reader)

        for i, row in tqdm(enumerate(reader)):
            text = row[2]
            hadm_id = row[1]
            subject_id = row[0]
            labels = row[3].split(';')

            # for each text, label pair, calculate heighest 
            # weighted n-gram in text
            for label in labels:
                myList = []

                # subject id
                myList.append(subject_id)
                # hadm id
                myList.append(hadm_id)

                # augmented coefficients matrix has dims (5000, 51918)
                # (num. labels, size vocab.)
                # get row corresponding to label:
                word_weights = mat_full[c2ind[label]]

                # get each set of n grams in text
                # get ngrams
                fourgrams = nltk.ngrams(text.split(), ngram)
                fourgrams_scores = []
                for grams in fourgrams:
                    # calculate score
                    sum_weights = 0
                    for word in grams:
                        if word in w2ind:
                            inx = w2ind[word]
                            # add coeff from logistic regression
                            # matrix for given word
                            sum_weights = sum_weights + word_weights[inx]
                        else:
                            # else if word not in vocab, adds 0 weight
                            pass
                    fourgrams_scores.append(sum_weights)

                # get the fourgram itself
                w = [word for word in text.split()][
                    fourgrams_scores.index(max(fourgrams_scores)):
                    fourgrams_scores.index(max(fourgrams_scores))+ngram]

                # label
                myList.append(label)
                # start index of 4-gram
                myList.append(fourgrams_scores.index(max(fourgrams_scores)))
                # 4-gram
                myList.append(" ".join(w))
                # sum weighted score (highest)
                myList.append(max(fourgrams_scores))
                writer.writerow(myList)
            if i % 3000 == 0:
                logging.info("Processed row {}: {}; my list: {}".format(
                    i, row, myList))
    f.close()


def main(args):

    # STEP 1: DATA PREPARATION

    # to handle large csv files
    csv.field_size_limit(sys.maxsize)
    if args.number_labels == '50':
        args.number_labels = 50
    train_bows_file = args.train_file.split('.csv')[0] + '_bows.csv'
    dev_bows_file = args.test_file.split('.csv')[0] + '_bows.csv'

    logging.info("Start log reg with BOW document representation")

    dicts = data_utils.load_lookups(
        args.train_file, args.model, args.number_labels, args.vocab)
    # w2ind: length 51917, {'000cc': 1, '000mcg': 2, '000mg': 3, '000s': 4, ..
    # 'conronary': 12656, 'cons': 12657, 'consciosness': 12658,..

    # c2ind: length 50, {'038.9': 0, '244.9': 1, '250.00': 2, '272.0': 3,
    #   '272.4': 4, '276.1': 5, '276.2': 6, '285.1': 7, '285.9': 8, '287.5': 9,
    #   '305.1': 10, '311': 11, '33.24': 12, '36.15': 13, '37.22': 14, ..
    #    ..., '995.92': 46, 'V15.82': 47, 'V45.81': 48, 'V58.61': 49}
    w2ind, ind2c, c2ind = dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    logging.info("Done with lookups, length of dicts: {}".format(len(dicts)))

    logging.info("Start log reg. Prepare or read in BOW data from: {}".format(
        train_bows_file))

    if not os.path.isfile(train_bows_file):

        # outputs:
        #   csr_matrix((data, indices, subj_inds)), np.array(yy), hadm_ids
        X, y, hadm_ids = construct_X_Y(
            args.train_file, args.number_labels, w2ind, c2ind)
        X_dv, y_dv, hadm_ids_dv = construct_X_Y(
            args.test_file, args.number_labels, w2ind, c2ind)

        # write out
        write_bows(train_bows_file, X, hadm_ids, y, ind2c)
        write_bows(dev_bows_file, X_dv, hadm_ids_dv, y_dv, ind2c)

        # read in bows files
        # train first two examples:
        # HADM_ID,BOW,LABELS
        # 182396,
        # 1765:1 3569:2 3660:1 3829:1 4406:1 4888:1 5446:4 8400:1 9541:1
        # 46919:4 <....> 1160:1 51264:4 51735:1 51765:1,  # 79 words total
        #   287.5;45.13;584.9
    else:
        logging.info("Found existing BOWs file: {}. Skip its "
                     "preprocessing".format(train_bows_file))
        X, yy_tr, hids_tr = read_bows(train_bows_file, c2ind)
        X_dv, yy_dv, hids_dv = read_bows(dev_bows_file, c2ind)

    # X.shape: (8066, 51918)
    # yy_tr.shape: (8066, 50)
    # X_dv.shape: (1573, 51918)
    # yy_dv.shape: (1573, 50)
    logging.info("X.shape: " + str(X.shape))
    logging.info("yy_tr.shape: " + str(yy_tr.shape))
    logging.info("X_dv.shape: " + str(X_dv.shape))
    logging.info("yy_dv.shape: " + str(yy_dv.shape))

    # deal with labels that don't have any positive examples
    # drop empty columns from yy. keep track of which columns kept
    # predict on test data with those columns. guess 0 on the others
    logging.info("Remove training labels witout positive examples")
    labels_with_examples = yy_tr.sum(axis=0).nonzero()[0]
    yy = yy_tr[:, labels_with_examples]
    logging.info(
        "Training labels with examples: {}, original:{}".format(
            yy.shape, yy_tr.shape))

    # STEP 2: ONE-VS-REST MULTILABEL CLASSIFICATION
    # build the classifier
    model = args.model
    if model == 'log_reg':
        logging.info("Building One-vs-Rest Log Reg classifier, with probs")
        # n_jobs=-1 means using all CPU resources
        # try solver=newton-cholesky, class_weight='balanced'
        clf = OneVsRestClassifier(LogisticRegression(
                C=args.c, max_iter=args.max_iter, penalty='l2', dual=False,
                solver='sag', class_weight='balanced',
                verbose=True, n_jobs=-1), n_jobs=-1)
    elif model == 'svc':
        logging.info("Building RBF SVC classifier with probabilities")
        # kernel='rbf' -- slow; TODO: more memory!!
        clf = OneVsRestClassifier(SVC(
                C=1, kernel='linear', class_weight='balanced',
                probability=True, decision_function_shape='ovr',
                cache_size=1000, verbose=True), n_jobs=-1)
    elif model == 'linear_svc':
        logging.info("Building Linear SVC classifier, no probs")
        clf = OneVsRestClassifier(LinearSVC(
                penalty='l2', loss='squared_hinge', dual=False, C=1,
                class_weight='balanced', verbose=True), n_jobs=-1)
    elif model == 'sgd_linear_svm_':
        logging.info("Building SGDClassifier with SGD training, no probs")
        clf = OneVsRestClassifier(SGDClassifier(
                penalty='l2', loss='hinge', dual=False, C=1,
                class_weight='balanced', verbose=True), n_jobs=-1)
    else:
        logging.error('Unsupported classifier: {}'.format(model))
    # TODO where is clf.coef_ ?
    # logging.info("clf.estimators_: {}".format(clf.estimators_))

    # train
    logging.info("Training/fitting classifier...")
    clf.fit(X, yy)

    logging.info("Predicting on dev set...")
    # yhat: binary predictions matrix
    # yhat_raw: prediction scores matrix (floats)
    yhat = clf.predict(X_dv)

    if any([model == 'svc', model == 'log_reg']):
        yhat_raw = clf.predict_proba(X_dv)
    else:
        yhat_raw = yhat
    logging.info(
        "Predicting on dev set. Example from yhat: {}".format(yhat[0]))

    # deal with labels that don't have positive training examples
    logging.info("Reshaping output to deal with labels missing from train set")
    labels_with_examples = set(labels_with_examples)
    yhat_full = np.zeros(yy_dv.shape)
    yhat_full_raw = np.zeros(yy_dv.shape)
    j = 0
    for i in range(yhat_full.shape[1]):
        if i in labels_with_examples:
            yhat_full[:, i] = yhat[:, j]
            yhat_full_raw[:, i] = yhat_raw[:, j]
            j += 1

    # evaluate
    # Precision@5 for top 50 labels experiment; @8/15 for 'full'
    logging.info("Evaluating on dev set...")
    k = 5 if args.number_labels == 50 else [8, 15]

    # only Logistic Regression and SCV provide probabilities
    # try to put yhat_full_raw to nothing or yhat_full
    if any([not model == 'svc', not model == 'log_reg']):
        yhat_full_raw = yhat_full
    metrics = evaluation.all_metrics(
        yhat_full, yy_dv, k=k, yhat_raw=yhat_full_raw)

    logging.info("metrics after dev evaluation: {}".format(metrics))
    evaluation.print_metrics(metrics)

    # save metric history, model, params
    model_dir = os.path.join(MODEL_DIR, '_'.join(
        [model, time.strftime('%Y%m%d_%H%M%S', time.localtime())]))
    os.mkdir(model_dir)
    if model == 'log_reg':
        logging.info("Saving predictions")
        preds_file = tools.write_preds(
            yhat_full, model_dir, hids_dv, 'test', yhat_full_raw)

    logging.info("Predicting on train set...")
    # yhat: binary predictions matrix
    # yhat_raw: prediction scores matrix (floats)
    yhat_tr = clf.predict(X)
    if any([model == 'svc', model == 'log_reg']):
        yhat_tr_raw = clf.predict_proba(X)
    else:
        yhat_tr_raw = yhat_tr

    # reshape output again
    logging.info("reshape output again...")
    yhat_tr_full = np.zeros(yy_tr.shape)
    yhat_tr_full_raw = np.zeros(yy_tr.shape)
    j = 0
    for i in range(yhat_tr_full.shape[1]):
        if i in labels_with_examples:
            yhat_tr_full[:, i] = yhat_tr[:, j]
            yhat_tr_full_raw[:, i] = yhat_tr_raw[:, j]
            j += 1
    logging.info(
        "reshaped yhat_tr_full: {}, yhat_tr_full_raw: {}".format(
            yhat_tr_full.shape, yhat_tr_full_raw.shape))

    # evaluate again
    logging.info("Evaluating on training set")
    if any([not model == 'svc', not model == 'log_reg']):
        yhat_tr_full_raw = None
    metrics_tr = evaluation.all_metrics(
        yhat_tr_full, yy_tr, k=k, yhat_raw=yhat_tr_full_raw)
    logging.info("metrics after train evaluation: {}".format(metrics))
    evaluation.print_metrics(metrics_tr)

    if args.ngram > 0:
        logging.info("Calculating {}-grams using file: {}".format(
            args.ngram, dev_bows_file))
        calculate_top_ngrams(
            dev_bows_file, clf, c2ind, w2ind, labels_with_examples, args.ngram)

    # Commenting this out because the models are huge (11G for mimic3 full)
    # logging.info("saving model")
    # with open("%s/model.pkl" % model_dir, 'wb') as f:
    #     pickle.dump(clf, f)

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
