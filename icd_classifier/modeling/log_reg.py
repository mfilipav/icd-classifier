"""
    Reads (or writes) BOW-formatted notes and performs scikit-learn logistic regression
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
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from icd_classifier.settings import MODEL_DIR, DATA_DIR
from icd_classifier.data import utils
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
        7908, 182396, admission date discharge date date of birth sex m service 
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
    if args.Y == '50':
        args.Y = 50

    logging.info("Start log reg")

    dicts = utils.load_lookups(args)
    # w2ind: length 51917, {'000cc': 1, '000mcg': 2, '000mg': 3, '000s': 4, ..
    # 'conronary': 12656, 'cons': 12657, 'consciosness': 12658,..

    # c2ind: length 50, {'038.9': 0, '244.9': 1, '250.00': 2, '272.0': 3,
    #   '272.4': 4, '276.1': 5, '276.2': 6, '285.1': 7, '285.9': 8, '287.5': 9,
    #   '305.1': 10, '311': 11, '33.24': 12, '36.15': 13, '37.22': 14, ..
    #    ..., '995.92': 46, 'V15.82': 47, 'V45.81': 48, 'V58.61': 49}
    w2ind, ind2c, c2ind = dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    logging.info("Done with lookups, length of dicts: {}".format(len(dicts)))

    # outputs: csr_matrix((data, indices, subj_inds)), np.array(yy), hadm_ids
    X, y, hadm_ids = construct_X_Y(
        args.train_file, args.Y, w2ind, c2ind)
    X_dv, y_dv, hadm_ids_dv = construct_X_Y(
        args.dev_file, args.Y, w2ind, c2ind)

    # write out
    train_bows = args.train_file.split('.csv')[0] + '_bows.csv'
    dev_bows = args.dev_file.split('.csv')[0] + '_bows.csv'
    write_bows(train_bows, X, hadm_ids, y, ind2c)
    write_bows(dev_bows, X_dv, hadm_ids_dv, y_dv, ind2c)

    # read in bows files
    # train first two examples:
    # HADM_ID,BOW,LABELS
    # 182396, 1765:1 3569:2 3660:1 3829:1 4406:1 4888:1 5446:4 8400:1 9541:1 9566:1 11221:1 11353:2 12337:1
    # 14086:2 14227:3 14250:1 14955:1 15097:1 15389:1 15394:1 15751:1 18270:1 19603:1 19690:2 20086:1 20581:1
    # 21407:1 22437:1 22444:1 22590:1 22916:1 23207:2 23441:1 24086:1 24631:1 25394:1 25769:1
    # 25959:1 26543:1 26816:2 27470:1 27737:1 28723:2 28988:1 29291:1 29447:1 30955:1 31318:2
    # 31320:1 31337:1 31441:1 32516:1 32697:2 33097:5 33163:1 33274:1 34284:1 34846:1 34914:2
    # 34943:1 37393:1 37408:1 40369:1 40557:1 42455:1 42836:1 42920:1 43518:1 46159:1 46919:4
    # 47072:1 47163:1 47565:2 48764:1 51035:1 51160:1 51264:4 51735:1 51765:1,
    #   287.5;45.13;584.9
    # 183363, 3569:4 4406:1 4634:1 5446:1 8400:1 9015:1 9269:2 9541:2 13739:1 14086:1
    # 14227:3 14243:1 14291:1 14362:1 14391:1 15227:1 15389:1 15538:1 15751:2 15970:1
    # 16433:1 17617:1 18554:1 18702:1 19587:1 20086:1 20581:1 20710:1 20853:1 22322:1
    # 22437:3 22440:1 22831:2 22833:1 23004:2 23207:1 23424:1 26183:2 26816:2 27470:2
    # 28988:1 29291:1 29447:1 30955:1 31318:3 31320:1 31339:2 31827:1 32697:2 33097:2
    # 33163:1 33250:1 33370:1 33535:1 33858:1 34943:1 38518:1 41586:2 42181:1 42451:1
    # 42836:2 42920:1 43129:1 45144:1 45157:1 46159:1 46919:4 47072:1 47565:2 48268:1
    # 50840:5 51160:1 51264:3 51735:1,
    #  272.4;401.9;96.71
    X, yy_tr, hids_tr = read_bows(train_bows, c2ind)
    X_dv, yy_dv, hids_dv = read_bows(dev_bows, c2ind)

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

    # STEP 2: ONE-VS-REST MULTILABEL LOGISTIC REGRESSION CLASSIFICATION

    # build the classifier
    logging.info("Building One-vs-Rest Log Reg classifier")
    # n_jobs=-1 means using all CPU resources
    clf = OneVsRestClassifier(
        LogisticRegression(
            C=args.c, max_iter=args.max_iter, solver='sag'), n_jobs=-1)
    # TODO where is clf.coef_ ?
    # logging.info("clf.estimators_: {}".format(clf.estimators_))

    # train
    logging.info("Training/fitting classifier...")
    #
    clf.fit(X, yy)

    # predict
    logging.info("predicting...")
    yhat = clf.predict(X_dv)
    yhat_raw = clf.predict_proba(X_dv)

    # deal with labels that don't have positive training examples
    logging.info("reshaping output to deal with labels missing from train set")
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
    logging.info("evaluating...")
    k = 5 if args.Y == 50 else [8, 15]
    metrics = evaluation.all_metrics(
        yhat_full, yy_dv, k=k, yhat_raw=yhat_full_raw)
    logging.info("metrics: {}".format(metrics))
    evaluation.print_metrics(metrics)

    # save metric history, model, params
    logging.info("saving predictions")
    model_dir = os.path.join(MODEL_DIR, '_'.join(
        ["log_reg", time.strftime('%Y%m%d_%H%M%S', time.localtime())]))
    os.mkdir(model_dir)
    preds_file = tools.write_preds(
        yhat_full, model_dir, hids_dv, 'test', yhat_full_raw)

    logging.info("sanity check on train")
    yhat_tr = clf.predict(X)
    yhat_tr_raw = clf.predict_proba(X)

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
    logging.info("evaluating again...")
    metrics_tr = evaluation.all_metrics(
        yhat_tr_full, yy_tr, k=k, yhat_raw=yhat_tr_full_raw)
    logging.info("metrics again: {}".format(metrics))
    evaluation.print_metrics(metrics_tr)

    if args.ngram > 0:
        logging.info("calculating {}-grams using file: {}".format(
            args.ngram, dev_bows))
        calculate_top_ngrams(
            dev_bows, clf, c2ind, w2ind, labels_with_examples, args.ngram)

    # Commenting this out because the models are huge (11G for mimic3 full)
    # logging.info("saving model")
    # with open("%s/model.pkl" % model_dir, 'wb') as f:
    #     pickle.dump(clf, f)

    logging.info("saving metrics")
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
    parser.add_argument("--data_path", type=str, help="path to a file ")
    parser.add_argument("--train_file", type=str, help="path to a file ")
    parser.add_argument("--dev_file", type=str, help="path to a file ")
    parser.add_argument(
        "--vocab", type=str,
        help="path to a file holding vocab word list for discretizing words")
    parser.add_argument(
        "--model", type=str, default='log_reg',
        help="model name is log reg, just for record keeping")
    parser.add_argument(
        "--Y", type=str, help="size of label space, 'full' or '50'")
    parser.add_argument("--ngram", type=int, default=0, help="size if ngrams")
    parser.add_argument(
        "--c", type=float, default=1.0, help="log reg parameter C")
    parser.add_argument(
        "--max_iter", type=int, default=20, help="log reg max iterations")

    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)
