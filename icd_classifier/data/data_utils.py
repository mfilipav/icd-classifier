from collections import defaultdict
import csv
import numpy as np
import logging
from icd_classifier.settings import DATA_DIR, MIMIC_3_DIR, MAX_LENGTH


def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude
        them.
        Generally, procedure codes have dots after the first two digits,
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code


def load_code_descriptions():
    # load description lookup from the appropriate data files
    desc_dict = defaultdict(str)
    with open("%s/D_ICD_DIAGNOSES.csv" % (DATA_DIR), 'r') as descfile:
        r = csv.reader(descfile)
        # header
        next(r)
        for row in r:
            code = reformat(row[1], True)
            desc = row[-1]
            desc_dict[code] = desc
        logging.debug(
            "1. Done loading ICD Diagnoses into desc_dict, "
            "dict length={}. Last item: {}:{}".format(
                len(desc_dict), code, desc))

    with open("%s/D_ICD_PROCEDURES.csv" % (DATA_DIR), 'r') as procfile:
        r = csv.reader(procfile)
        # header
        next(r)
        for row in r:
            code = row[1]
            desc = row[-1]
            if code not in desc_dict.keys():
                code = reformat(code, False)
                desc_dict[code] = desc
        logging.debug(
            "2. Done loading ICD Procedures into desc_dict, "
            "dict length={}. Last item: {}:{}".format(
                len(desc_dict), code, desc))

    with open('%s/ICD9_descriptions' % DATA_DIR, 'r') as labelfile:
        for row in labelfile:
            row = row.rstrip().split()
            code = row[0]
            if code not in desc_dict.keys():
                desc = ' '.join(row[1:])
                desc_dict[code] = desc
        logging.debug(
            "3. Done loading ICD Descriptions into desc_dict, "
            "dict length={}. Last item: {}:{}".format(
                len(desc_dict), code, desc))
    logging.info("Done. Size of desc_dict: {}".format(len(desc_dict)))
    return desc_dict


def load_description_vectors(Y):
    # load description one-hot vectors from file
    dv_dict = {}
    data_dir = MIMIC_3_DIR
    with open("%s/description_vectors.vocab" % (data_dir), 'r') as vfile:
        r = csv.reader(vfile, delimiter=" ")
        # header
        next(r)
        for row in r:
            code = row[0]
            vec = [int(x) for x in row[1:]]
            dv_dict[code] = vec
    return dv_dict


def pad_desc_vecs(desc_vecs):
    # logging.info(
    #    "Padding description vectors to have same length in a batch")
    desc_len = max([len(dv) for dv in desc_vecs])
    pad_vecs = []
    for vec in desc_vecs:
        if len(vec) < desc_len:
            vec.extend([0] * (desc_len - len(vec)))
        pad_vecs.append(vec)
    return pad_vecs


class Batch:
    """
    Prepares a batch of data. Numpy arrays for each data row with text
    (idx of word tokens), labels (multi-hot-encoded label indices), hadmis
    Size of Batch is 16 by default

    Returns:
        docs: list of len batch_size, contains idxs of word tokens,
            padded to max sequence length in batch or `MAX_LENGTH`
        labels: multihot-encoded labels (ICD-9 code)
        label_idx_set: set of label indices that are found in the batch
    """
    def __init__(self, desc_embed):
        self.docs = []
        self.labels = []
        self.hadm_ids = []
        self.label_idx_set = set()
        self.length = 0
        self.max_length = MAX_LENGTH
        self.desc_embed = desc_embed
        self.descs = []

    def add_example_to_batch(
            self, row, ind2c, c2ind, w2ind, dv_dict, num_labels):
        """
            Makes an instance to add to this batch from given row data,
            with a bunch of lookups
            row-derived: text, hadm_id, length
        """
        text = row[2]
        hadm_id = int(row[1])
        length = int(row[4])
        current_label_idx_set = set()
        # will store labels' indices
        labels_idx = np.zeros(num_labels)
        labelled = False

        # get codes as a multi-hot vector
        # row[3]: 287.5;45.13;584.9
        for code in row[3].split(';'):
            if code in c2ind.keys():
                label_idx = int(c2ind[code])
                labels_idx[label_idx] = 1
                current_label_idx_set.add(label_idx)
                labelled = True
        if not labelled:
            return

        desc_vecs = []
        if self.desc_embed:
            for idx in current_label_idx_set:
                code = ind2c[idx]
                if code in dv_dict.keys():
                    # need to copy or description padding will get screwed up
                    desc_vecs.append(dv_dict[code][:])
                else:
                    desc_vecs.append([len(w2ind)+1])

        # OOV words are given a unique index at end of vocab lookup
        text = [
            int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in text.split()
        ]

        # truncate long documents
        if len(text) > self.max_length:
            text = text[:self.max_length]
        # reset length
        self.length = min(self.max_length, length)

        # build instance
        self.docs.append(text)
        self.labels.append(labels_idx)
        self.hadm_ids.append(hadm_id)
        self.label_idx_set = self.label_idx_set.union(current_label_idx_set)
        if self.desc_embed:
            self.descs.append(pad_desc_vecs(desc_vecs))

    def pad_docs(self):
        # pad all docs to have self.length
        padded_docs = []
        for doc in self.docs:
            if len(doc) < self.length:
                doc.extend([0] * (self.length - len(doc)))
            padded_docs.append(doc)
        self.docs = padded_docs

    def to_return(self):
        # logging.debug(
        #     "hadm_ids: {}, len of one padded doc: {}, self labels: {}, "
        #     "label_idx_set: {}".format(
        #         self.hadm_ids, len(self.docs[0]),
        #         self.labels[0], self.label_idx_set))
        return np.array(self.docs), np.array(self.labels),\
            np.array(self.hadm_ids), self.label_idx_set, np.array(self.descs)


def data_generator(filename, dicts, batch_size, num_labels, desc_embed=False):
    """
        Inputs:
            filename: holds data sorted by sequence length, for batching
            dicts: holds all needed lookups
            batch_size: the batch size for train iterations
            num_labels: size of label output space
            desc_embed: true if using DR-CAML (lambda > 0)
        Yields:
            np arrays with data for training loop, see add_example_to_batch()
    """
    # logging.debug(
    #     "Create batch with {} np arrays with sorted-by-length data for "
    #     "training loop; data source: {} label space: {}".format(
    #         batch_size, filename, num_labels))

    # ind2w = dicts['ind2w']
    w2ind = dicts['w2ind']
    ind2c = dicts['ind2c']
    c2ind = dicts['c2ind']
    dv_dict = dicts['dv']
    with open(filename, 'r') as infile:
        csv_reader = csv.reader(infile)
        # header
        next(csv_reader)
        batch = Batch(desc_embed)
        for row in csv_reader:
            # find the next `batch_size` instances
            if len(batch.docs) == batch_size:
                batch.pad_docs()
                yield batch.to_return()
                # re-initialize the batch
                batch = Batch(desc_embed)
            # keep adding instance
            batch.add_example_to_batch(
                row, ind2c, c2ind, w2ind, dv_dict, num_labels)
        batch.pad_docs()
        yield batch.to_return()


def load_vocab_dict(model, number_labels, vocab_file, public_model=False):
    # reads vocab_file into two lookups (word:ind) and (ind:word)
    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for line in vocabfile:
            line = line.rstrip()
            if line != '':
                vocab.add(line.strip())
    # hack because the vocabs were created differently for these models
    if all([number_labels == 'full', public_model, model == 'conv_attn']):
        ind2w = {i: w for i, w in enumerate(sorted(vocab))}
    else:
        ind2w = {i + 1: w for i, w in enumerate(sorted(vocab))}
    w2ind = {w: i for i, w in ind2w.items()}
    return ind2w, w2ind


def load_full_codes(train_path):
    """
        Inputs:
            train_path: path to train dataset
        Outputs:
            code lookup, description lookup
    """
    # get description lookup
    desc_dict = load_code_descriptions()
    # build code lookups from appropriate datasets
    codes = set()
    for split in ['train', 'dev', 'test']:
        with open(train_path.replace('train', split), 'r') as f:
            lr = csv.reader(f)
            next(lr)
            for row in lr:
                for code in row[3].split(';'):
                    codes.add(code)
        logging.info(
            "Done loading code descriptions for split: {}".format(split))
    codes = set([c for c in codes if c != ''])
    ind2c = defaultdict(
        str, {i: c for i, c in enumerate(sorted(codes))})
    logging.info(
        "Done preparing code and description lookup. Example code ind2c: "
        "len={}, first5={}, last={}; example desc_dict: len={}, f5={}, "
        "last={}".format(
            len(ind2c),
            list(ind2c.items())[0:4],
            list(ind2c.items())[-1],
            len(desc_dict),
            list(desc_dict.items())[0:4],
            list(desc_dict.items())[-1])
        )

    return ind2c, desc_dict


def load_lookups(train_path, model, number_labels, vocab,
                 public_model=False, desc_embed=False):
    """
        Inputs:
        Outputs:
            vocab lookups, ICD code lookups, description lookup,
            description one-hot vector lookup
    """
    # get vocab lookups
    ind2w, w2ind = load_vocab_dict(model, number_labels, vocab, public_model)

    # get code and description lookups
    if number_labels == 'full':
        ind2c, desc_dict = load_full_codes(train_path)
    else:
        codes = set()
        with open(
            "%s/TOP_%s_CODES.csv" % (
                MIMIC_3_DIR, str(number_labels)), 'r') as labelfile:
            lr = csv.reader(labelfile)
            for row in lr:
                codes.add(row[0])
        ind2c = {i: c for i, c in enumerate(sorted(codes))}
        desc_dict = load_code_descriptions()
    c2ind = {c: i for i, c in ind2c.items()}

    # get description one-hot vector lookup
    if desc_embed:
        dv_dict = load_description_vectors(number_labels)
    else:
        dv_dict = None

    dicts = {
        'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c,
        'c2ind': c2ind, 'desc': desc_dict, 'dv': dv_dict}
    return dicts
