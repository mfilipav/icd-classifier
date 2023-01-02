import torch
import csv
import json
import numpy as np

from icd_classifier.modeling import models
from icd_classifier.settings import *
from icd_classifier.data import data_utils
import logging


def pick_model(args, dicts):
    """
        Use args to initialize the appropriate model
    """
    number_labels = len(dicts['ind2c'])
    if args.model == "log_reg":
        model = models.LogReg(
            number_labels, args.embed_file, args.lmbda, args.gpu, dicts, args.pool,
            args.embed_size, args.dropout, args.code_emb)

    elif args.model == "basic_cnn":
        filter_size = int(args.filter_size)
        model = models.BasicCNN(
            number_labels, args.embed_file, filter_size, args.num_filter_maps, args.gpu,
            dicts, args.embed_size, args.dropout)

    elif args.model == "rnn":
        logging.info(
            type(number_labels), type(args.embed_file), type(dicts), type(args.rnn_dim),
            type(args.cell_type), type(args.rnn_layers), type(args.gpu),
            type(args.embed_size), type(args.bidirectional))
        model = models.VanillaRNN(
            number_labels, args.embed_file, dicts, args.rnn_dim, args.cell_type,
            args.rnn_layers, args.gpu, args.embed_size, args.bidirectional)

    elif args.model == "conv_attn":
        filter_size = int(args.filter_size)
        model = models.ConvAttnPool(
            number_labels, args.embed_file, filter_size, args.num_filter_maps,
            args.lmbda, args.gpu, dicts, embed_size=args.embed_size,
            dropout=args.dropout, code_emb=args.code_emb)

    else:
        # rewrite with "try - except" pattern
        logging("ERROR: unknown model '{}'".format(args.model))

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
        args.number_labels, args.filter_size, args.dropout, args.num_filter_maps,
        args.rnn_dim, args.cell_type, args.rnn_layers, args.lmbda,
        args.command, args.weight_decay, args.data_path,
        args.vocab, args.embed_file, args.lr]
    param_names = [
        "number_labels", "filter_size", "dropout", "num_filter_maps", "rnn_dim",
        "cell_type", "rnn_layers", "lmbda", "command",
        "weight_decay", "data_path", "vocab", "embed_file", "lr"]
    params = {
        name: val for name, val in zip(
            param_names, param_vals) if val is not None}
    return params


def build_code_vecs(code_inds, dicts):
    """
        Get vocab-indexed arrays representing words in 
        descriptions of each *unseen* label
    """
    code_inds = list(code_inds)
    ind2w, ind2c, dv_dict = dicts['ind2w'], dicts['ind2c'], dicts['dv']
    vecs = []
    for c in code_inds:
        code = ind2c[c]
        if code in dv_dict.keys():
            vecs.append(dv_dict[code])
        else:
            # vec is a single UNK if not in lookup
            vecs.append([len(ind2w) + 1])
    # pad everything
    vecs = data_utils.pad_desc_vecs(vecs)
    return (torch.cuda.LongTensor(code_inds), vecs)


def save_metrics(metrics_hist_all, model_dir):
    with open(model_dir + "/metrics.json", 'w') as metrics_file:
        # concatenate dev, train metrics into one dict
        data = metrics_hist_all[0].copy()
        data.update({"%s_te" % (name):val for (name, val) in metrics_hist_all[1].items()})
        data.update({"%s_tr" % (name):val for (name, val) in metrics_hist_all[2].items()})
        json.dump(data, metrics_file, indent=1)


def save_params_dict(params):
    with open(params["model_dir"] + "/params.json", 'w') as params_file:
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
                eval_val = np.nanargmin(metrics_hist_all[0][early_stopping_metric])
            else:
                eval_val = np.nanargmax(metrics_hist_all[0][early_stopping_metric])

            if eval_val == len(metrics_hist_all[0][early_stopping_metric]) - 1:
                # save state dict
                sd = model.cpu().state_dict()
                torch.save(sd, model_dir + "/model_best_%s.pth" % early_stopping_metric)
                if args.gpu:
                    model.cuda()
    print("saved metrics, params, model to directory %s\n" % (model_dir))
