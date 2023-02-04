from collections import defaultdict
import csv
import numpy as np
from tqdm import tqdm
import logging
from nltk.tokenize import RegexpTokenizer
from icd_classifier.settings import (
    MIMIC_3_DIR, DESCRIPTIONS_DIAGNOSES_FILE,
    DESCRIPTIONS_PROCEDURES_FILE,
    DESCRIPTIONS_CODES_FILE,
    DESCRIPTIONS_VECTORS_FILE, MAX_LENGTH)


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


def load_code_descriptions(descriptions_diagnoses_file,
                           descriptions_procedures_file,
                           descriptions_codes_file):
    """
    load ICD9 code description lookup from 3 descriptions files

    Diagnoses example:
    row_id, icd9_code, short_title, long_title
    1, 01716, Erythem nod tb-oth test,
        "Erythema nodosum with hypersensitivity reaction in tuberculosis,
        tubercle bacilli not found by bacteriological or histological
        examination, but tuberculosis confirmed by other methods
        [inoculation of animals]"
    2, 01720, TB periph lymph-unspec,
        "Tuberculosis of peripheral lymph nodes, unspecified"
    9757,9596,Hip & thigh injury NOS,Hip and thigh injury --> '959.6'
    9869,E8981,Fire accident --> 'E898.1'

    Procedures example:
    row_id, icd9_code, short_title, long_title
    1, 1423, Chorioret les xenon coag,
        Destruction of chorioretinal lesion by xenon arc photocoagulation
    7, 1431, Retinal tear diathermy, Repair of retinal tear by diathermy

    Codes example:
    @	ICD9 Hierarchy Root
    00	Procedures and interventions, Not Elsewhere Classified
    00-99.99	PROCEDURES
    00.0	Therapeutic ultrasound
    00.01	Therapeutic ultrasound of vessels of head and neck
    ..
    00.1	Pharmaceuticals
    00.10	Implantation of chemotherapeutic agent
    00.11	Infusion of drotrecogin alfa (activated)
    00.12	Administration of inhaled nitric oxide
    ..
    00.2	Intravascular imaging of blood vessels
    00.21	Intravascular imaging of extracranial cerebral vessels
    00.22	Intravascular imaging of intrathoracic vessels
    ..
    00.3	Computer assisted surgery [CAS]
    00.31	Computer assisted surgery with CT/CTA
    00.32	Computer assisted surgery with MR/MRA
    ..
    00.4	Adjunct Vascular System Procedures
    00.40	Adjunct vascular system procedure on single vessel
    00.41	Adjunct vascular system procedure on two vessels
    ..
    00.5	Other cardiovascular procedures
    00.50	Implantation of cardiac resynchronization pacemaker without mention
        of defibrillation, total system [CRT-P]
    00.51	Implantation of cardiac resynchronization defibrillator, total
        system [CRT-D]
    ..
    00.9	Other procedures and interventions
    00.91	Transplant from live related donor
    00.92	Transplant from live non-related donor
    00.93	Transplant from cadaver
    00.94	Intra-operative neurophysiologic monitoring
    001	Cholera
    001-009.99	INTESTINAL INFECTIOUS DISEASES
    001-139.99	INFECTIOUS AND PARASITIC DISEASES
    001-999.99	DISEASES AND INJURIES
    001.0	Cholera due to Vibrio cholerae
    001.1	Cholera due to Vibrio cholerae el tor
    ..
    E800	Railway accident involving collision with rolling stock
    E800-E807.9	RAILWAY ACCIDENTS
    E800-E999.9	SUPPLEMENTARY CLASSIFICATION OF EXTERNAL CAUSES OF INJURY
        AND POISONING
    E800.0	Railway accident involving collision with rolling stock and
        injuring railway employee
    ..
    V01	Contact with or exposure to communicable diseases
    V01-V06.99	PERSONS WITH POTENTIAL HEALTH HAZARDS RELATED TO COMMUNICABLE
        DISEASES
    V01-V86.99	SUPPLEMENTARY CLASSIFICATION OF FACTORS INFLUENCING HEALTH
        STATUS AND CONTACT WITH HEALTH SERVICES
    V01.0	Contact with or exposure to cholera
    V01.1	Contact with or exposure to tuberculosis


    Output:
    dict with {'code': 'code description'} structure
    V87.32:Contact with and (suspected) exposure to algae bloom
    91.89:Microscopic examination of specimen from other site, other
        microscopic examination
    V86.1:ESTROGEN RECEPTOR STATUS
    Size of 'desc_dict': 22,267 codes
        14,567 from diagnoses, 3,857 from procedures and 3,843 from codes file
    """
    desc_dict = defaultdict(str)

    with open(descriptions_diagnoses_file, 'r') as diagnoses_f:
        r = csv.reader(diagnoses_f)
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

    with open(descriptions_procedures_file, 'r') as procedures_f:
        r = csv.reader(procedures_f)
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

    with open(descriptions_codes_file, 'r') as codes_f:
        for row in codes_f:
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


def vocab_index_descriptions(
        vocab_file, vectors_file):
    """
    Pre-computes the vocab-indexed version of each code description
    Output:
    CODE VECTOR
    017.20 48630 33097 35465 28649 32143 49435
        (desc: Tuberculosis of peripheral lymph nodes, unspecified examination)
    ..
    V86 18583 39582 44551
        (desc: Estrogen receptor status)
    V86-V86.99 18583 39582 44551
        (desc: ESTROGEN RECEPTOR STATUS)
    """
    # load lookups
    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i, line in enumerate(vocabfile):
            line = line.strip()
            if line != '':
                vocab.add(line)
    ind2w = {i+1: w for i, w in enumerate(sorted(vocab))}
    w2ind = {w: i for i, w in ind2w.items()}
    desc_dict = load_code_descriptions(
        DESCRIPTIONS_DIAGNOSES_FILE,
        DESCRIPTIONS_PROCEDURES_FILE,
        DESCRIPTIONS_CODES_FILE)

    tokenizer = RegexpTokenizer(r'\w+')

    with open(vectors_file, 'w') as output_file:
        w = csv.writer(output_file, delimiter=' ')
        w.writerow(["CODE", "VECTOR"])
        for code, desc in tqdm(desc_dict.items()):
            # same tokenizing as in data.get_discharge_summaries()
            tokens = [
                t.lower() for t in tokenizer.tokenize(
                    desc) if not t.isnumeric()]
            inds = [
                w2ind[t] if t in w2ind.keys() else len(
                    w2ind) + 1 for t in tokens]
            # code: 017.20
            # desc: Tuberculosis of peripheral lymph nodes, unspecified
            # tokens: [
            #   'tuberculosis', 'of', 'peripheral', 'lymph',
            #   'nodes', 'unspecified']
            # inds: [48630, 33097, 35465, 28649, 32143, 49435]
            # logging.debug("code: {}, desc: {}, tokens: {}, inds: {}".format(
            #     code, desc, tokens, inds))
            w.writerow([code] + [str(i) for i in inds])


def load_description_vectors(descriptions_vectors_file):
    # load description one-hot vectors from file
    dv_dict = {}
    with open(descriptions_vectors_file, 'r') as vfile:
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


def load_codes_and_descriptions(train_path, number_labels):
    """
        Inputs:
            train_path: path to train dataset
            number_labels: 'full' or int with top n labels
        Outputs:
            code lookup, description lookup
    """
    # get description lookup
    desc_dict = load_code_descriptions(
        DESCRIPTIONS_DIAGNOSES_FILE,
        DESCRIPTIONS_PROCEDURES_FILE,
        DESCRIPTIONS_CODES_FILE)

    # build code lookups from appropriate datasets
    codes = set()

    if number_labels == 'full':

        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    for code in row[3].split(';'):
                        codes.add(str(code))
            logging.info(
                "Done loading code descriptions for split: {}".format(split))
        codes = set([c for c in codes if c != ''])

    else:
        with open(
            "%s/TOP_%s_CODES.csv" % (
                MIMIC_3_DIR, str(number_labels)), 'r') as labelfile:
            lr = csv.reader(labelfile)
            for row in lr:
                codes.add(row[0])

    ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})

    # this is new, to deal with the fact that in XML clf
    logging.info("Codes in ind2c: {}, {}".format(type(ind2c), len(ind2c)))
    bad_codes = []
    for item in ind2c.copy().items():
        code = item[1]
        key = item[0]
        found_value = desc_dict.get(code)
        if found_value is None:
            logging.warning('code: {} not found in desc_dict'.format(code))
            bad_codes.append(item)
            ind2c.pop(key)
    logging.info("Codes in ind2c: {}".format(len(ind2c)))
    logging.info("Codes from ind2c not found in desc_dict: {}, "
                 "codes: {}".format(len(bad_codes), bad_codes))

    logging.info(
        "Done preparing code and description lookup for number_labels: {}. "
        "Example code ind2c: "
        "len={}, first5={}, last={}; \nExample desc_dict: len={}, f5={}, "
        "last={}".format(
            number_labels,
            len(ind2c),
            list(ind2c.items())[0:4],
            list(ind2c.items())[-1],
            len(desc_dict),
            list(desc_dict.items())[0:4],
            list(desc_dict.items())[-1])
    )

    return ind2c, desc_dict


def load_lookups(
        train_path, model, number_labels, vocab,
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
    ind2c, desc_dict = load_codes_and_descriptions(train_path, number_labels)
    c2ind = {c: i for i, c in ind2c.items()}

    # get description one-hot vector lookup
    if desc_embed:
        dv_dict = load_description_vectors(DESCRIPTIONS_VECTORS_FILE)
    else:
        dv_dict = None

    dicts = {
        'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c,
        'c2ind': c2ind, 'desc': desc_dict, 'dv': dv_dict}
    return dicts
