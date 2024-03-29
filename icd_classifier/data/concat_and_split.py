"""
    Concatenate the labels with the notes data and split using the saved splits
"""
import csv
import logging
from icd_classifier.settings import MIMIC_3_DIR

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def concat_data(labelsfile, notes_file, outfilename):
    """
        INPUTS:
            labelsfile: sorted by hadm id, contains one label per line
            notes_file: sorted by hadm id, contains one note per line
    """
    with open(labelsfile, 'r') as lf:
        logging.info(
            f"Concatenating labels file: {labelsfile} and notes files: "
            "{notes_file} into one file: {outfilename}")
        with open(notes_file, 'r') as notesfile:
            # outfilename = '%s/notes_labeled.csv' % MIMIC_3_DIR
            with open(outfilename, 'w') as outfile:
                w = csv.writer(outfile)
                w.writerow(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])

                labels_gen = next_labels(lf)
                notes_gen = next_notes(notesfile)

                for i, (subj_id, text, hadm_id) in enumerate(notes_gen):
                    if i % 10000 == 0:
                        logging.info(str(i) + " done")
                    _, cur_labels, cur_hadm = next(labels_gen)

                    if cur_hadm == hadm_id:
                        w.writerow([
                            subj_id, str(hadm_id), text, ';'.join(cur_labels)])
                    else:
                        logging.error(
                            "cur_hadm={} doesn't match with hadm_id={}, data "
                            "is probably sorted incorrectly!".format(
                                cur_hadm, hadm_id))
                        break
    return outfilename


def split_data(labeledfile, base_name):
    logging.info('Splitting data')

    # create and write headers for train, dev, test
    train_name = '%s_train_split.csv' % (base_name)
    dev_name = '%s_dev_split.csv' % (base_name)
    test_name = '%s_test_split.csv' % (base_name)
    train_file = open(train_name, 'w')
    dev_file = open(dev_name, 'w')
    test_file = open(test_name, 'w')
    train_file.write(','.join(
        ['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    dev_file.write(','.join(
        ['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    test_file.write(','.join(
        ['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    logging.info('train split name: ', train_name)
    logging.info('labeled name: ', labeledfile)

    hadm_ids = {}

    # read in train, dev, test splits
    for splt in ['train', 'dev', 'test']:
        hadm_ids[splt] = set()
        splt_hadm_ids = '%s/%s_full_hadm_ids.csv' % (MIMIC_3_DIR, splt)
        logging.info("reading in from files: {}".format(splt_hadm_ids))
        with open(splt_hadm_ids, 'r') as f:
            for line in f:
                hadm_ids[splt].add(line.rstrip())

    with open(labeledfile, 'r') as lf:
        reader = csv.reader(lf)
        next(reader)
        for i, row in enumerate(reader):
            # filter text, write to file according to train/dev/test split
            hadm_id = row[1]
            if i % 10000 == 0:
                logging.debug(
                    "Row {} read. HADM_ID = {}.\nRow: {}".format(
                        str(i), hadm_id, row))

            # catch entries without labels
            labels = row[3]
            if labels:
                if hadm_id in hadm_ids['train']:
                    train_file.write(','.join(row) + "\n")
                elif hadm_id in hadm_ids['dev']:
                    dev_file.write(','.join(row) + "\n")
                elif hadm_id in hadm_ids['test']:
                    test_file.write(','.join(row) + "\n")
            else:
                logging.warning(f"Missing labels at row {i}, row: {row}")
                continue

        train_file.close()
        dev_file.close()
        test_file.close()
    logging.info("Done splitting. Train file name: {}".format(train_name))

    return train_name, dev_name, test_name


def next_labels(labelsfile):
    """
        Generator for label sets from the label file
    """
    labels_reader = csv.reader(labelsfile)
    # header
    next(labels_reader)

    first_label_line = next(labels_reader)

    cur_subj = int(first_label_line[0])
    cur_hadm = int(first_label_line[1])
    cur_labels = [first_label_line[2]]

    for row in labels_reader:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        code = row[2]
        # keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_labels, cur_hadm
            cur_labels = [code]
            cur_subj = subj_id
            cur_hadm = hadm_id
        else:
            # add to the labels and move on
            cur_labels.append(code)
    yield cur_subj, cur_labels, cur_hadm


def next_notes(notesfile):
    """
        Generator for notes from the notes file
        This will also concatenate discharge summaries and their addenda,
        which have the same subject and hadm id
    """
    nr = csv.reader(notesfile)
    # header
    next(nr)

    first_note = next(nr)

    cur_subj = int(first_note[0])
    cur_hadm = int(first_note[1])
    cur_text = first_note[3]

    for row in nr:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        text = row[3]
        # keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_text, cur_hadm
            cur_text = text
            cur_subj = subj_id
            cur_hadm = hadm_id
        else:
            # concatenate to the discharge summary and move on
            cur_text += " " + text
    yield cur_subj, cur_text, cur_hadm
