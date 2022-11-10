import os
from collections import Counter
import pandas as pd
import pickle
import numpy as np
import logging
from icd_classifier.settings import Keys, Tables, DATA_DIR, DATA_COMPRESSION_GZ


class Dataset:
    """
    Creates a dataset for MIMIC-III database discharge summaries with
    corresponding diagnosis codes.

    """

    def __init__(self):
        if DATA_COMPRESSION_GZ:
            compression = 'gzip'
        else:
            compression = 'infer'
        icd = pd.read_csv(Tables.diagnoses_icd, compression=compression,
                            on_bad_lines='skip')
        
        logging.info('number of ICDs: {}'.format(len(icd)))
        assert len(icd) == 651047

        # takes a while
        notes = pd.read_csv(Tables.note_events, compression=compression,
                            on_bad_lines='skip', nrows='full')

        # Only get discharge notes
        discharge_notes = notes[notes[Keys.note_category] ==
                                "Discharge summary"]

        # transform discharge notes into a list

        dat = []
        for index, row in discharge_notes.iterrows():
            logging.debug(
                "index: {}, icd: {}, Keys.admission_id: {}, row: {}".format(
                    index, icd, Keys.admission_id, row))
            one_row = [
                row[Keys.text],
                (icd[icd[Keys.admission_id] == row[Keys.admission_id]])[
                    Keys.icd9].astype(str).tolist()
            ]
            dat.append(one_row)
        # list comprehension way, should be faster
        # dat = [[row[Keys.text],
        #         (icd[icd[Keys.admission_id] ==
        #             row[Keys.admission_id]])
        #         [Keys.icd9].astype(str).tolist()]
        #         for index, row in discharge_notes.iterrows()]

