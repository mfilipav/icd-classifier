"""
    Reads NOTEEVENTS file, finds the discharge summaries,
    preprocesses them and writes out the filtered dataset.
"""
import csv
import gzip
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

# retain only alphanumeric
tokenizer = RegexpTokenizer(r'\w+')


def write_discharge_summaries(notes_file, out_file):
    # notes_file = '%s/NOTEEVENTS.csv.gz' % (MIMIC_3_DIR)
    print("processing notes file")

    with gzip.open(notes_file, 'rt',) as csvfile:
        # notes_file = notes_file.split('.gz')[0]
        # with open(notes_file, 'r') as csvfile:
        with open(out_file, 'w') as outfile:
            print("writing to %s" % (out_file))
            outfile.write(','.join([
                'SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']) + '\n')
            notereader = csv.reader(csvfile)
            # header
            next(notereader)
            i = 0
            for line in tqdm(notereader):
                subj = int(line[1])
                category = line[6]
                if category == "Discharge summary":
                    note = line[10]
                    #tokenize, lowercase and remove numerics
                    tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
                    text = '"' + ' '.join(tokens) + '"'
                    outfile.write(','.join([line[1], line[2], line[4], text]) + '\n')
                i += 1
    return out_file