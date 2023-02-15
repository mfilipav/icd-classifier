import torch
import csv
import json
import numpy as np
import pandas as pd
from icd_classifier.modeling import models
from icd_classifier.settings import DATA_DIR
from icd_classifier.data import data_utils
import logging
from sklearn.preprocessing import MultiLabelBinarizer


def pick_model(args, dicts):
    """
        Use args to initialize the appropriate model
    """
    logging.info("Picking model: {}".format(args.model))
    number_labels = len(dicts['ind2c'])
    if args.model == "basic_cnn":
        filter_size = int(args.filter_size)
        model = models.BasicCNN(
            number_labels, args.embeddings_file, filter_size, args.filter_maps,
            args.gpu,
            dicts, args.embedding_size, args.dropout)

    elif args.model == "rnn":
        model = models.RNN(
            number_labels, args.embeddings_file, dicts, args.rnn_dim,
            args.rnn_cell_type, args.rnn_layers, args.dropout, args.gpu,
            args.batch_size, args.embedding_size, args.bidirectional)

    elif args.model == "caml":
        filter_size = int(args.filter_size)
        model = models.CAML(
            number_labels, args.embeddings_file, filter_size, args.filter_maps,
            args.lmbda, args.gpu, dicts, embedding_size=args.embedding_size,
            dropout=args.dropout)
    else:
        # rewrite with "try - except" pattern
        logging.error("ERROR: unknown model '{}'".format(args.model))

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)

    if args.gpu:
        model.cuda()

    return model


def make_param_dict(args):
    """
        Make a list of parameters to save for future reference
    """
    param_vals = [
        args.number_labels, args.filter_size, args.dropout, args.filter_maps,
        args.rnn_dim, args.rnn_cell_type, args.rnn_layers, args.lmbda,
        args.command, args.weight_decay, args.data_path,
        args.vocab, args.embeddings_file, args.lr]
    param_names = [
        "number_labels", "filter_size", "dropout", "filter_maps", "rnn_dim",
        "rnn_cell_type", "rnn_layers", "lmbda", "command",
        "weight_decay", "data_path", "vocab", "embeddings_file", "lr"]
    params = {
        name: val for name, val in zip(
            param_names, param_vals) if val is not None}
    return params


def build_code_vecs(code_inds, dicts):
    """
        Get vocab-indexed arrays representing words in
        descriptions of each *unseen* label
    """
    logging.info("Building code vectors")
    code_inds = list(code_inds)
    ind2w, ind2c, dv_dict = dicts['ind2w'], dicts['ind2c'], dicts['dv']
    vecs = []
    for c in code_inds:
        code = ind2c[c]
        if code in dv_dict.keys():
            vecs.append(dv_dict[code])
        else:
            # vec is a single UNK token if not in lookup
            vecs.append([len(ind2w) + 1])
    # pad everything
    vecs = data_utils.pad_desc_vecs(vecs)
    long_tensor_code_inds = torch.cuda.LongTensor(code_inds)
    logging.info(
        "Done building code vectors. Shape code_inds: {}, its tensor: {}, "
        "vecs: {}".format(
            len(code_inds), long_tensor_code_inds.shape, len(vecs)))

    return (long_tensor_code_inds, vecs)


def save_metrics(metrics_hist_all, model_dir):
    metrics_file = model_dir + "/metrics.json"
    logging.info("Saving metrics to: {}".format(metrics_file))
    with open(metrics_file, 'w') as metrics_file:
        # concatenate dev, train metrics into one dict
        data = metrics_hist_all[0].copy()
        data.update(
            {"%s_te" % (name):
                val for (name, val) in metrics_hist_all[1].items()})
        data.update(
            {"%s_tr" % (name):
                val for (name, val) in metrics_hist_all[2].items()})
        json.dump(data, metrics_file, indent=1)


def save_params_dict(params):
    params_file = params["model_dir"] + "/params.json"
    logging.info("Saving params to: {}".format(params_file))
    with open(params_file, 'w') as params_file:
        json.dump(params, params_file, indent=1)


def write_preds(yhat, model_dir, hids, fold, ind2c, yhat_raw=None):
    """
        INPUTS:
            yhat: binary predictions matrix
            model_dir: which directory to save in
            hids: list of hadm_id's to save along with predictions
            fold: train, dev, or test
            ind2c: code lookup
            yhat_raw: predicted scores matrix (floats)
    """
    preds_file = "%s/preds_%s.psv" % (model_dir, fold)
    with open(preds_file, 'w') as f:
        w = csv.writer(f, delimiter='|')
        for yhat_, hid in zip(yhat, hids):
            codes = [ind2c[ind] for ind in np.nonzero(yhat_)[0]]
            if len(codes) == 0:
                w.writerow([hid, ''])
            else:
                w.writerow([hid] + list(codes))
    if fold != 'train' and yhat_raw is not None:
        # write top 100 scores so we can re-do @k metrics later
        # top 100 only - saving the full set of scores
        # is very large (~1G for mimic-3 full test set)
        scores_file = '%s/pred_100_scores_%s.json' % (model_dir, fold)
        scores = {}
        sortd = np.argsort(yhat_raw)[:, ::-1]
        for i, (top_idxs, hid) in enumerate(zip(sortd, hids)):
            scores[int(hid)] = {
                ind2c[idx]:
                    float(yhat_raw[i][idx]) for idx in top_idxs[:100]}
        with open(scores_file, 'w') as f:
            json.dump(scores, f, indent=1)

    logging.info("Saving predictions as {}".format(preds_file))

    return preds_file


def save_everything(args, metrics_hist_all, model, model_dir,
                    params, early_stopping_metric, evaluate=False):
    """
        Save metrics, model, params all in model_dir
    """
    save_metrics(metrics_hist_all, model_dir)
    params['model_dir'] = model_dir
    save_params_dict(params)

    if not evaluate:
        # save the model with the best early_stopping_metric metric
        if not np.all(np.isnan(metrics_hist_all[0][early_stopping_metric])):
            if early_stopping_metric == 'loss_dev':
                eval_val = np.nanargmin(
                    metrics_hist_all[0][early_stopping_metric])
            else:
                eval_val = np.nanargmax(
                    metrics_hist_all[0][early_stopping_metric])

            if eval_val == len(metrics_hist_all[0][early_stopping_metric]) - 1:
                # save state dict
                sd = model.cpu().state_dict()
                best_model = model_dir+"/model_best_%s.pth" % \
                    early_stopping_metric
                logging.info(
                    "Save best model to file: {}, evaluated with: {}".format(
                        best_model, early_stopping_metric))
                torch.save(sd, best_model)
                if args.gpu:
                    model.cuda()
    logging.info("Saved metrics, params, model to directory: {}".format(
        model_dir))


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
    logging.info(
        "Last 5 data items before conversion: {}, and after: {}"
        "".format(data[-5:], converted_data[-5:]))

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
    backup_code = None
    for item in c2ind.items():
        code = item[0]
        value = desc_dict.get(code)
        if value is not None:
            code_to_desc_dict[code] = desc_dict.get(code)
            backup_code = code
        else:
            code_to_desc_dict[code] = desc_dict.get(backup_code)
            logging.error(
                f"No desc for code: {code}, c2ind: {item}. Replace it with"
                f" the previous code: {backup_code}")
    return code_to_desc_dict


def prepare_x_z_corpus_files(
        train_df, test_df, code_to_desc_dict, path_x_trn,
        path_x_tst, path_z, path_corpus):
    dfz = pd.DataFrame.from_dict(
        code_to_desc_dict, orient='index',
        columns=['description'], dtype='object')
    dfz.iloc[:, 0].to_csv(
        path_or_buf=path_z, index=False, header=False)
    logging.info(
        f"Prepared label description file {path_z}, line number "
        "corresponds to code in c2ind")
    train_df.iloc[:, 2].to_csv(
        path_or_buf=path_x_trn, index=False, header=False)
    logging.info(f"Prepared training text file: {path_x_trn}")
    test_df.iloc[:, 2].to_csv(
        path_or_buf=path_x_tst, index=False, header=False)
    logging.info(f"Prepared testing/dev text file: {path_x_tst}")

    # concate Z and X.trn into joint corpus
    # cat ./X.trn.txt Z.all.txt > pecos_50/full_corpus.txt
    corpus = pd.concat(
        [dfz.iloc[:, 0], train_df.iloc[:, 2]], axis=0, join='outer')
    corpus.to_csv(
        path_or_buf=path_corpus, index=False, header=False)
    logging.info(
        f"Prepared a joint training text and label \
        description corpus: {path_corpus}")


def encode_y_labels(train_df, test_df, label_name, c2ind,
                    ind_list, prepare_text_files, number_labels):
    # prepare list of all labels:
    # get labels for datesets into a list of tuples
    logging.info("get label tuples for train data")
    labels_tr = get_label_tuples(
        train_df, label_name, return_idx=True, c2ind=c2ind)
    logging.info("get label tuples for test data")
    labels_te = get_label_tuples(
        test_df, label_name, return_idx=True, c2ind=c2ind)

    if prepare_text_files:
        path_y_trn = DATA_DIR+'/Y.trn.'+str(number_labels)+'.txt'
        path_y_tst = DATA_DIR+'/Y.tst.'+str(number_labels)+'.txt'

        with open(path_y_trn, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(labels_tr)
        with open(path_y_tst, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(labels_te)

    # create multihot label encoding, CSR matrix
    logging.info("Preparing CSR matrices for Y train, test")
    label_encoder_multilabel = MultiLabelBinarizer(
        classes=ind_list, sparse_output=True)
    Y_trn = label_encoder_multilabel.fit_transform(labels_tr)
    Y_tst = label_encoder_multilabel.fit_transform(labels_te)

    # cast as correct dtype
    Y_trn = Y_trn.astype(dtype=np.float32, copy=False)
    Y_tst = Y_tst.astype(dtype=np.float32, copy=False)

    # Y_trn is a csr matrix with a shape (47719, 8887) and
    # 745363 non-zero values.
    logging.info(
        f"Y_trn is a {Y_trn.getformat()} matrix with a shape {Y_trn.shape} "
        f"and {Y_trn.nnz} non-zero values.")
    logging.info(
        f"Y_tst is a {Y_tst.getformat()} matrix with a shape {Y_tst.shape} "
        f"and {Y_tst.nnz} non-zero values.")

    return Y_trn, Y_tst
