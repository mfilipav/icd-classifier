"""
    Main training code. Loads data, builds the model, trains, tests,
    evaluates with many metrics, saves predictions
"""
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import logging
import csv
import argparse
import sys
import time
from tqdm import tqdm
from collections import defaultdict
import os  # need for args
from icd_classifier.settings import MODEL_DIR
from icd_classifier.data import data_utils
from icd_classifier.modeling import tools, evaluation, interpret


def main(args):
    start = time.time()
    args, model, optimizer, params, dicts = init(args)
    epochs_trained = train_epochs(args, model, optimizer, params, dicts)

    logging.info(
        "TOTAL ELAPSED TIME FOR %s MODEL AND %d EPOCHS: %f"
        "" % (args.model, epochs_trained, time.time() - start))


def init(args):
    """
        Load data, build model, create optimizer,
        create vars to hold metrics, etc.
    """
    # need to handle really large text fields
    csv.field_size_limit(sys.maxsize)

    # load vocab and other lookups
    logging.info("Initialize with ARGS: {}".format(args))

    logging.info("Loading lookups...")
    desc_embed = args.lmbda > 0
    dicts = data_utils.load_lookups(
        args.data_path, args.model, args.number_labels, args.vocab,
        args.public_model, desc_embed=desc_embed)

    model = tools.pick_model(args, dicts)
    logging.info("Picked model: {}".format(model))

    if not args.test_model:
        logging.info(
            "Training mode, test_model set to: {}. Will initialize optimizer "
            "for training".format(args.test_model))
        optimizer = optim.Adam(
            model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    else:
        logging.info(
            "Testing, test_model set to: {}. Will initialize optimizer for "
            "training".format(args.test_model))
        optimizer = None

    params = tools.make_param_dict(args)

    return args, model, optimizer, params, dicts


def early_stop(metrics_hist, early_stopping_metric, patience):
    if not np.all(np.isnan(metrics_hist[early_stopping_metric])):
        length = len(metrics_hist[early_stopping_metric])
        if length >= patience:
            logging.info(
                "check for early stop, because patience={} is exceeded with:"
                " {}. Criterion={}".format(
                    patience, length, early_stopping_metric))
            if early_stopping_metric == 'loss_dev':
                return np.nanargmin(metrics_hist[early_stopping_metric]) < len(
                        metrics_hist[early_stopping_metric]) - patience
            else:
                return np.nanargmax(metrics_hist[early_stopping_metric]) < len(
                        metrics_hist[early_stopping_metric]) - patience
    else:
        # keep training if early_stopping_metric has no results yet
        return False


def train(model, optimizer, epoch, batch_size, data_path,
          gpu, dicts, quiet):
    """
        Training loop.
        output: losses for each example for this iteration
    """
    logging.info("EPOCH %d" % epoch)
    num_labels = len(dicts['ind2c'])

    losses = []
    # how often to print some info to stdout
    print_every = 25

    ind2c = dicts['ind2c']
    unseen_code_inds = set(ind2c.keys())
    desc_embed = model.lmbda > 0

    model.train()
    generator = data_utils.data_generator(
        data_path, dicts, batch_size, num_labels, desc_embed=desc_embed)

    for batch_idx, tup in tqdm(enumerate(generator)):
        data, target_labels, _, code_set, descs = tup
        data, target_labels = Variable(
            torch.LongTensor(data)), Variable(torch.FloatTensor(target_labels))
        unseen_code_inds = unseen_code_inds.difference(code_set)
        if gpu:
            data = data.cuda()
            target_labels = target_labels.cuda()
        optimizer.zero_grad()

        if desc_embed:
            desc_data = descs
        else:
            desc_data = None

        output, loss, _ = model(data, target_labels, desc_data=desc_data)

        loss.backward()
        optimizer.step()

        # losses.append(loss.data[0])
        losses.append(loss.data.item())

        if not quiet and batch_idx % print_every == 0:
            # print the average loss of the last 10 batches
            logging.info(
                "Train epoch: {} [batch #{}, batch_size {}, padded seq "
                "length {}]\t Loss: {:.6f},".format(
                    epoch, batch_idx, data.size()[0], data.size()[1],
                    np.mean(losses[-10:])))
    return losses, unseen_code_inds


def one_epoch(model, optimizer, number_labels, epoch, n_epochs, batch_size,
              data_path, testing, dicts, model_dir,
              save_tp_fp_examples, gpu, quiet):
    """
        Wrapper to do a training epoch and test on dev
    """
    if not testing:
        losses, unseen_code_inds = train(
            model, optimizer, epoch, batch_size,
            data_path, gpu, dicts, quiet)
        loss = np.mean(losses)
        logging.info("epoch {} loss: {}".format(epoch, loss))
    else:
        loss = np.nan
        if model.lmbda > 0:
            # still need to get unseen code inds
            logging.info(
                "model.lmbda is non-zero: {}, getting set of codes not in "
                "training set".format(model.lmbda))
            c2ind = dicts['c2ind']
            unseen_code_inds = set(dicts['ind2c'].keys())
            num_labels = len(dicts['ind2c'])
            logging.debug(
                "num labels: {}".format(num_labels))
            with open(data_path, 'r') as f:
                r = csv.reader(f)
                # header
                next(r)
                for row in r:
                    unseen_code_inds = unseen_code_inds.difference(
                        set([c2ind[c] for c in row[3].split(';') if c != '']))
            logging.info(
                "num codes not in train set: %d" % len(unseen_code_inds))
        else:
            unseen_code_inds = set()

    fold = 'dev'
    if epoch == n_epochs - 1:
        logging.info("Reached last epoch: {}! Will test on test "
                     "and train sets".format(epoch))
        testing = True
        quiet = False

    # test on dev
    metrics = test(
        model, number_labels, epoch, data_path, fold, gpu, unseen_code_inds,
        dicts, save_tp_fp_examples, model_dir, testing)
    if testing or epoch == n_epochs - 1:
        logging.info("Evaluating on test")
        metrics_te = test(
            model, number_labels, epoch, data_path, "test",
            gpu, unseen_code_inds,
            dicts, save_tp_fp_examples, model_dir, True)
    else:
        metrics_te = defaultdict(float)

    metrics_tr = {'loss': loss}
    metrics_all = (metrics, metrics_te, metrics_tr)
    logging.info("Training metrics: {}".format(metrics_tr))
    logging.info("Testing metrics_te: {}".format(metrics_te))
    logging.info("Testing metrics: {}".format(metrics))

    return metrics_all


def train_epochs(args, model, optimizer, params, dicts):
    """
        Main loop. Does train on many epochs and test
    """

    logging.info(
        "Main loop of training for {} epochs".format(args.n_epochs))
    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])

    test_only = args.test_model is not None
    evaluate = args.test_model is not None
    # train for n_epochs unless early_stopping_metric does
    # not improve for [patience] epochs
    for epoch in range(args.n_epochs):
        # only test on train/test set on very last epoch
        if epoch == 0 and not args.test_model:
            model_dir = os.path.join(
                MODEL_DIR, '_'.join(
                    [args.model, time.strftime(
                        '%Y%m%d_%H%M%S', time.localtime())]
                )
            )
            os.mkdir(model_dir)
        elif args.test_model:
            model_dir = os.path.dirname(
                os.path.abspath(args.test_model))

        metrics_all = one_epoch(
            model, optimizer, args.number_labels, epoch, args.n_epochs,
            args.batch_size, args.data_path, test_only, dicts, model_dir,
            args.save_tp_fp_examples, args.gpu, args.quiet)

        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])

        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])

        for name in metrics_all[2].keys():
            metrics_hist_tr[name].append(metrics_all[2][name])

        metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)

        # save metrics, model, params
        tools.save_everything(
            args, metrics_hist_all, model, model_dir,
            params, args.early_stopping_metric, evaluate)

        if test_only:
            # we're done
            break

        if args.early_stopping_metric in metrics_hist.keys():
            if early_stop(
                    metrics_hist, args.early_stopping_metric, args.patience):
                # stop training, do tests on test and train sets,
                # and then stop the script
                logging.info(
                    "%s hasn't improved in %d epochs, early stopping..." % (
                        args.early_stopping_metric, args.patience))
                test_only = True
                args.test_model = '%s/model_best_%s.pth' % (
                    model_dir, args.early_stopping_metric)
                model = tools.pick_model(args, dicts)
                logging.info("Will test model: {}".format('model'))
    return epoch + 1


def unseen_code_vecs(model, code_inds, dicts, gpu):
    """
        Use description module for codes not seen in training set.
    """
    code_vecs = tools.build_code_vecs(code_inds, dicts)
    code_inds, vecs = code_vecs
    # wrap it in an array so it's 3d
    desc_embeddings = model.embed_descriptions([vecs], gpu)[0]
    # replace relevant final_layer weights with desc embeddings 
    model.final.weight.data[code_inds, :] = desc_embeddings.data
    model.final.bias.data[code_inds] = 0


def test(model, number_labels, epoch, data_path, fold, gpu, code_inds,
         dicts, save_tp_fp_examples, model_dir, testing):
    """
        Testing loop.
        Returns metrics
    """
    filename = data_path.replace('train', fold)
    logging.info('Testing file: %s' % filename)
    num_labels = len(dicts['ind2c'])
    ind2c = dicts['ind2c']

    # initialize stuff for saving attention samples
    if save_tp_fp_examples:
        tp_file = open(
            '%s/tp_%s_examples_%d.txt' % (model_dir, fold, epoch), 'w')
        fp_file = open(
            '%s/fp_%s_examples_%d.txt' % (model_dir, fold, epoch), 'w')
        window_size = model.conv.weight.data.size()[2]

    y, yhat, yhat_raw, hids, losses = [], [], [], [], []

    desc_embed = model.lmbda > 0
    if desc_embed and len(code_inds) > 0:
        unseen_code_vecs(model, code_inds, dicts, gpu)

    model.eval()
    one_batch = data_utils.data_generator(
        filename, dicts, 1, num_labels, desc_embed=desc_embed)
    for batch_idx, tup in tqdm(enumerate(one_batch)):
        data, target_labels, hadm_ids, _, descs = tup
        # data, target_labels = Variable(torch.LongTensor(data), volatile=True),
        # Variable(torch.FloatTensor(target_labels))
        # TODO: do we need to use torch.no_grad()??
        data = torch.LongTensor(data)
        target_labels = torch.FloatTensor(target_labels)

        if gpu:
            data = data.cuda()
            target_labels = target_labels.cuda()
        model.zero_grad()

        if desc_embed:
            desc_data = descs
        else:
            desc_data = None

        # get an attention sample for 2% of batches
        get_attention = save_tp_fp_examples and (
            np.random.rand() < 0.02 or (fold == 'test' and testing))
        output, loss, alpha = model(
            data, target_labels, desc_data=desc_data,
            get_attention=get_attention)

        # output = F.sigmoid(output)
        output = torch.sigmoid(output)
        output = output.data.cpu().numpy()
        # losses.append(loss.data[0])
        losses.append(loss.data.item())
        target_labels_data = target_labels.data.cpu().numpy()
        if get_attention and save_tp_fp_examples:
            interpret.save_samples(
                data, output, target_labels_data, alpha, window_size,
                epoch, tp_file, fp_file, dicts=dicts)

        # save predictions, target_labels, hadm ids
        yhat_raw.append(output)
        output = np.round(output)
        y.append(target_labels_data)
        yhat.append(output)
        hids.extend(hadm_ids)

    # close files if needed
    if save_tp_fp_examples:
        tp_file.close()
        fp_file.close()

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    # write the predictions
    preds_file = tools.write_preds(
        yhat, model_dir, hids, fold, ind2c, yhat_raw)
    logging.info("Wrote predictions into file: {}".format(preds_file))

    # get metrics
    k = 5 if num_labels == 50 else [8, 15]

    metrics = evaluation.all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    evaluation.print_metrics(metrics)
    metrics['loss_%s' % fold] = np.mean(losses)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, help="path to a file "
        "containing sorted train data. dev/test splits assumed to have same "
        "name format with 'train' replaced by 'dev' and 'test'")
    parser.add_argument(
        "--vocab", type=str,
        help="path to a file holding vocab word list for discretizing words")
    parser.add_argument(
        "--number_labels", type=str,
        help="size of label space, full or 50")
    parser.add_argument(
        "--model", type=str,
        choices=["basic_cnn", "rnn", "conv_attn", "multi_conv_attn",
                 "log_reg", "saved"],
        help="model")
    parser.add_argument(
        "--n_epochs", type=int, help="number of epochs to train")
    parser.add_argument(
        "--embeddings_file", type=str, required=False,
        help="path to a file holding pre-trained embeddings")
    parser.add_argument(
        "--rnn_cell_type", type=str, choices=["lstm", "gru"], default='gru',
        help="what kind of RNN to use (default: GRU)")
    parser.add_argument(
        "--rnn_dim", type=int, required=False, default=128,
        help="size of rnn hidden layer (default: 128)")
    parser.add_argument(
        "--rnn_bidirectional", dest="bidirectional", action="store_true",
        required=False, default=False,
        help="optional flag for rnn to use a bidirectional model")
    parser.add_argument(
        "--rnn_layers", type=int, required=False, default=1,
        help="number of layers for RNN models (default: 1)")
    parser.add_argument(
        "--embedding_size", type=int, required=False, default=100,
        help="size of embedding dimension. (default: 100)")
    parser.add_argument(
        "--filter_size", type=str, required=False, dest="filter_size",
        default=4, help="size of convolution filter to use, if multi_conv_attn"
                        ", give comma-separated e.g. 3,4,5")
    parser.add_argument(
        "--filter_maps", type=int, required=False, default=50,
        help="number of maps, i.e, size of conv output (default: 50)")
    parser.add_argument(
        "--codes_embeddings", type=str, required=False,
        help="code embeddings file used for param initialization")
    parser.add_argument(
        "--weight_decay", type=float, required=False, default=0,
        help="coeff for l2 norm of model weights")
    parser.add_argument(
        "--lr", type=float, required=False, dest="lr", default=1e-3,
        help="learning rate for Adam optimizer (default=1e-3)")
    parser.add_argument("--batch_size", type=int, required=False, default=16,
                        help="number of examples per training batch")
    parser.add_argument(
        "--dropout", dest="dropout", type=float, required=False, default=0.5,
        help="optional specification of dropout (default: 0.5)")
    parser.add_argument(
        "--lmbda", type=float, required=False, default=0,
        help="param for tradeoff between BCE and similarity embedding losses, "
             "if 0, won't create/use the description embedding module at all")
    parser.add_argument(
        "--test_model", type=str, dest="test_model", required=False,
        help="path to a saved model to load and evaluate")
    parser.add_argument(
        "--early_stopping_metric", type=str, default='f1_micro', choices=[
            'acc_macro', 'prec_macro', 'rec_macro', 'f1_macro', 'acc_micro',
            'prec_micro', 'rec_micro', 'f1_micro', 'rec_at_5', 'prec_at_5',
            'f1_at_5', 'auc_macro', 'auc_micro'],
        required=False, dest="early_stopping_metric",
        help="which metric to use for early stopping (default: f1_micro)")
    parser.add_argument(
        "--patience", type=int, default=5, required=False, dest="patience",
        help="how many epochs to wait for improved early_stopping_metric "
             "before early stopping")
    parser.add_argument(
        "--gpu", dest="gpu", action="store_const", required=False, const=True,
        help="optional flag to use GPU if available")
    parser.add_argument(
        "--public_model", action="store_const", required=False, const=True,
        default=False, help="optional flag for testing pre-trained models "
                            "from the public github")
    parser.add_argument(
        "--stack_filters", action="store_const", required=False, const=True,
        help="optional flag for multi_conv_attn to instead use concatenated "
             "filter outputs, rather than pooling over them")
    parser.add_argument(
        "--save_tp_fp_examples", action="store_const", required=False,
        const=True,
        help="optional flag to save samples of good / bad predictions")
    parser.add_argument(
        "--quiet", action="store_const", required=False, const=True,
        help="optional flag not to print so much during training")

    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)
