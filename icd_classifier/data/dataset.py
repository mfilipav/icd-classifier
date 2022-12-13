import os
from collections import Counter
import pandas as pd
import pickle
import logging
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from icd_classifier.settings import Keys, Tables, DATA_DIR, DATA_COMPRESSION_GZ


class Dataset:
    """
    Creates a dataset for MIMIC-III database discharge summaries with
    corresponding diagnosis codes. Performs train-test split with 90-10
    percentage. Persists the processed dataset as pkl file.

    Parameters
    ----------
    nrows : int
        Number of discharge records to be read from MIMIC-III notes. If None
        all records are considered.

    augment_labels : bool
        Set it to True if label list for each document should be augmented by
        the parents of each label. ICD9 code hierarchy provided by this repo
        https://github.com/sirrice/icd9

    top_n : int
        If provided, filters out labels such that only top n labels in terms
        of frequency remains

    Attributes
    ----------
    X : list
        List of medical discharge records from MIMIC-III database
    X_train: list
        Training split of medical discharge records from MIMIC-III database
    X_test: list
        Test split of medical discharge records from MIMIC-III database
    y_raw: list
        List of lists of ICD9 diagnosis codes corresponding to each document
    y_train_raw: list
        Train split of list of lists of ICD9 diagnosis codes corresponding
        to each document
    y_test_raw: list
        Test split of list of lists of ICD9 diagnosis codes corresponding
        to each document
    y_train: ndarray
        Hot-encoded version of y_train_raw
    y_test: ndarray
        Hot-encoded version of y_train_raw
    binarizer: sklearn.preprocessing.MultiLabelBinarizer
        binarizer which is used to transform the label lists into hot encoded
        versions.
    labels: list
        list of labels where each entry corresponds to the index in the
        y_train or y_test

    Notes
    -----

    If the test split includes ICD codes that are not included in the training
    set, those labels are automatically removed from the test set since there
    is no sample to support inference.
    """

    def __init__(self, nrows=None, augment_labels=False, top_n=None):
        
        nrows_str = str(nrows) if nrows is not None else "full"
        if augment_labels:
            rows_str += "_augmented"
        if top_n is not None:
            nrows_str += "_" + str("top_n")

        logging.info('Initializing Dataset object. Params:\n'
            '  - augmented_labels: {}\n'
            '  - top_n: {}'.format(augment_labels, top_n))

        path = os.path.join(DATA_DIR, "data_{}.pkl".format(nrows_str))
        
        if not os.path.isfile(path):
            logging.info(
                'First time loading dataset, did not find any files at: {}\n'
                'Data will be read from gzip files: {}'.format(
                    path, DATA_COMPRESSION_GZ))
            if DATA_COMPRESSION_GZ:
                compression = 'gzip'
            else:
                compression = 'infer'

            # show all columns            
            pd.set_option('display.max_columns', None)

            icd = pd.read_csv(Tables.diagnoses_icd, compression=compression,
                on_bad_lines='skip')   
            
            # 65,1047
            logging.info('number of ICDs: {}'.format(len(icd)))
            #   Column      Non-Null Count   Dtype
            # ---  ------      --------------   -----
            #  0   ROW_ID      651047 non-null  int64
            #  1   SUBJECT_ID  651047 non-null  int64
            #  2   HADM_ID     651047 non-null  int64
            #  3   SEQ_NUM     651000 non-null  float64
            #  4   ICD9_CODE   651000 non-null  object
            logging.info('icd df info: {}'.format(icd.info(verbose=True)))
            #    ROW_ID  SUBJECT_ID  HADM_ID  SEQ_NUM ICD9_CODE
            # 0    1297         109   172335      1.0     40301
            # 1    1298         109   172335      2.0       486
            # 2    1299         109   172335      3.0     58281
            # 3    1300         109   172335      4.0      5855
            # 4    1301         109   172335      5.0      4254
            # 5    1302         109   172335      6.0      2762
            # 6    1303         109   172335      7.0      7100
            # 7    1304         109   172335      8.0      2767
            # 8    1305         109   172335      9.0      7243
            # 9    1306         109   172335     10.0     45829
            logging.info('first 10 ICDs:\n{}'.format(icd[:10]))
            #         ROW_ID  SUBJECT_ID  HADM_ID  SEQ_NUM ICD9_CODE
            # 651037  639793       97497   168949      6.0      5178
            # 651038  639794       97497   168949      7.0     42731
            # 651039  639795       97497   168949      8.0     V5861
            # 651040  639796       97497   168949      9.0     45829
            # 651041  639797       97503   188195      1.0      7842
            # 651042  639798       97503   188195      2.0     20280
            # 651043  639799       97503   188195      3.0     V5869
            # 651044  639800       97503   188195      4.0     V1279
            # 651045  639801       97503   188195      5.0      5275
            # 651046  639802       97503   188195      6.0      5569
            logging.info('last 10 ICDs:\n{}'.format(icd[-10:]))

            notes = pd.read_csv(Tables.note_events, compression=compression,
                                on_bad_lines='skip', nrows=nrows)
            
            # 2,083,180
            logging.info('number of notes: {}'.format(len(notes)))
            #  #   Column       Dtype
            # ---  ------       -----
            #  0   ROW_ID       int64
            #  1   SUBJECT_ID   int64
            #  2   HADM_ID      float64
            #  3   CHARTDATE    object
            #  4   CHARTTIME    object
            #  5   STORETIME    object
            #  6   CATEGORY     object
            #  7   DESCRIPTION  object
            #  8   CGID         float64
            #  9   ISERROR      float64
            #  10  TEXT         object
            logging.info('notes df info: {}'.format(notes.info(verbose=True)))
            
            # what are other categories?
            # out of 2M notes only 59,6k are discharge summaries
            
            #    ROW_ID  SUBJECT_ID   HADM_ID   CHARTDATE CHARTTIME STORETIME
            # 0     174       22532  167853.0  2151-08-04       NaN       NaN
            # 1     175       13702  107527.0  2118-06-14       NaN       NaN
            # 2     176       13702  167118.0  2119-05-25       NaN       NaN
            # 3     177       13702  196489.0  2124-08-18       NaN       NaN
            # 4     178       26880  135453.0  2162-03-25       NaN       NaN
            # 5     179       53181  170490.0  2172-03-08       NaN       NaN
            # 6     180       20646  134727.0  2112-12-10       NaN       NaN
            # 7     181       42130  114236.0  2150-03-01       NaN       NaN
            # 8     182       56174  163469.0  2118-08-12       NaN       NaN
            # 9     183       56174  189681.0  2118-12-09       NaN       NaN
            #             CATEGORY DESCRIPTION  CGID  ISERROR  \
            # 0  Discharge summary      Report   NaN      NaN
            # 1  Discharge summary      Report   NaN      NaN
            # 2  Discharge summary      Report   NaN      NaN
            # 3  Discharge summary      Report   NaN      NaN
            # 4  Discharge summary      Report   NaN      NaN
            # 5  Discharge summary      Report   NaN      NaN
            # 6  Discharge summary      Report   NaN      NaN
            # 7  Discharge summary      Report   NaN      NaN
            # 8  Discharge summary      Report   NaN      NaN
            # 9  Discharge summary      Report   NaN      NaN
            #                                                 TEXT
            # 0  Admission Date:  [**2151-7-16**]       Dischar...
            # 1  Admission Date:  [**2118-6-2**]       Discharg...
            # 2  Admission Date:  [**2119-5-4**]              D...
            # 3  Admission Date:  [**2124-7-21**]              ...
            # 4  Admission Date:  [**2162-3-3**]              D...
            # 5  Admission Date:  [**2172-3-5**]              D...
            # 6  Admission Date:  [**2112-12-8**]              ...
            # 7  Admission Date:  [**2150-2-25**]              ...
            # 8  Admission Date:  [**2118-8-10**]              ...
            # 9  Admission Date:  [**2118-12-7**]              ...
            logging.info('first 10 notes:\n{}'.format(notes[:10]))
            #           ROW_ID  SUBJECT_ID   HADM_ID   CHARTDATE            CHARTTIME  \
            # 2083177  2070659       31097  115637.0  2132-01-21  2132-01-21 16:42:00
            # 2083178  2070660       31097  115637.0  2132-01-21  2132-01-21 18:05:00
            # 2083179  2070661       31097  115637.0  2132-01-21  2132-01-21 18:05:00
            #                    STORETIME       CATEGORY DESCRIPTION     CGID  ISERROR  \
            # 2083177  2132-01-21 16:44:00  Nursing/other      Report  20104.0      NaN
            # 2083178  2132-01-21 18:16:00  Nursing/other      Report  16023.0      NaN
            # 2083179  2132-01-21 18:31:00  Nursing/other      Report  16023.0      NaN
            #                                                       TEXT
            # 2083177  Family Meeting Note\nFamily meeting held with ...
            # 2083178  NPN 1800\n\n\n#1 Resp: [**Known lastname 2243*...
            # 2083179  NPN 1800\nNursing Addendum:\n[**Known lastname...
            logging.info('last 10 notes:\n{}'.format(notes[-10:]))

            # Only get discharge notes
            discharge_notes = notes[notes[Keys.note_category] ==
                                    "Discharge summary"]
            # 59652
            logging.info('number of discharge notes: {}'.format(len(discharge_notes)))
            logging.info('discharge notes df info: {}'.format(discharge_notes.info(verbose=True)))
            #    ROW_ID  SUBJECT_ID   HADM_ID   CHARTDATE CHARTTIME STORETIME  \
            # 0     174       22532  167853.0  2151-08-04       NaN       NaN                                                                                                                                     1     175       13702  107527.0  2118-06-14       NaN       NaN
            # 2     176       13702  167118.0  2119-05-25       NaN       NaN
            # ..
            # 60414   59622       66717  169165.0  2129-08-14       NaN       NaN
            # 60415   59623       73790  157100.0  2113-07-18       NaN       NaN
            logging.info('first 10 discharge notes:\n{}'.format(discharge_notes[:10]))
            logging.info('last 10 discharge notes:\n{}'.format(discharge_notes[-10:]))

            # dat = [[row[Keys.text],
            #         (icd[icd[Keys.admission_id] ==
            #             row[Keys.admission_id]])
            #         [Keys.icd9].astype(str).tolist()]
            #         for index, row in discharge_notes.iterrows()]

            dat = []
            logging.info('Start creating dat list of discharge notes')
            for index, row in discharge_notes.iterrows():
                # logging.debug("index: {}, icd: {}, Keys.admission_id: {}, row: {}".format(index, icd, Keys.admission_id, row))
                one_row = [
                    row[Keys.text],
                    (icd[icd[Keys.admission_id] == row[Keys.admission_id]])[ 
                        Keys.icd9].astype(str).tolist()
                ]
                dat.append(one_row)
                if index % 20000 == 0:
                    logging.info('  *** Example at index: {}\nrow: {}'.format(index, row))
                    # ['Admission Date:  [**2151-7-16**]       Discharge Date:  [**2151-8-4**]\n\n\nService:\nADDENDUM:\n\nRADIOLOGIC STUDIES:  Radiologic studies also included a chest\nCT, which confirmed cavitary les
                    # ions in the left lung apex\nconsistent with infectious process/tuberculosis.  This also\nmoderate-sized left pleural effusion.\n\nHEAD CT:  Head CT showed no intracranial hemorrhage or mass\neffec
                    # t, but old infarction consistent with past medical\nhistory.\n\nABDOMINAL CT:  Abdominal CT showed lesions of\nT10 and sacrum most likely secondary to osteoporosis. These can\nbe followed by repea
                    # t imaging as an outpatient.\n\n\n\n                            [**First Name8 (NamePattern2) **] [**First Name4 (NamePattern1) 1775**] [**Last Name (NamePattern1) **], M.D.  [**MD Number(1) 1776**
                    # ]\n\nDictated By:[**Hospital 1807**]\nMEDQUIST36\n\nD:  [**2151-8-5**]  12:11\nT:  [**2151-8-5**]  12:21\nJOB#:  [**Job Number 1808**]\n',
                    # ['01193', '4254', '42731', '2639', '2762', '5070', '5119', '2113']]
                    logging.info('  *** dat will be appended with:\n{}'.format(one_row))
            # 59652
            logging.info('Finished creating dat list of discharge notes. '
                'Size: {}'.format(len(dat)))
            
            # Prepare X, Y train-test datasets
            logging.info('Prepare X, Y train-test datasets')
            # Do not train embeddings with X, use X_train because it will
            # cause data snooping.
            self.X, self.y_raw = zip(*dat)
            # filter out top_n most frequent labels
            if top_n is not None: self._top_labels(top_n)
            if augment_labels: self._augment_labels()
            # Train: (53686, 53686)
            # Test: (5966, 5966)
            self._create_train_test_dataset()

            # transform list of labels to a (samples x classes)
            # binary matrix indicating the presence of a class label
            logging.info('Transform train-test labels into MultiLabel format')
            self.binarizer = MultiLabelBinarizer()
            self.y_train = self.binarizer.fit_transform(self.y_train_raw)
            self.y_test = self.binarizer.transform(self.y_test_raw)
            self.labels = self.binarizer.classes_
            
            logging.info('Finished transforming train-test labels into '
                'MultiLabel format')
            
            # create network graph
            self.y_train_graph = self._create_network_graph()

            # Pickle saving
            file = open(path, "wb")
            pickle.dump(self.X, file)
            pickle.dump(self.y_raw, file)
            pickle.dump(self.binarizer, file)
            pickle.dump(self.y_train_graph, file)

        else:
            logging.info('Loading the already processed and pickled dataset '
                'from: {}'.format(path))

            # Pickle loading
            file = open(path, "rb")
            logging.info('file object: {}'.format(file))
            self.X = pickle.load(file)
            self.y_raw = pickle.load(file)
            self.binarizer = pickle.load(file)
            self.y_train_graph = pickle.load(file)
            logging.info('Finished loading pickled samples, labels, '
                         'binarizer, graph')
            
            logging.info('Create train test dataset from samples and labels')
            self._create_train_test_dataset()
            logging.info('Transform train-test labels into MultiLabel format')
            self.y_train = self.binarizer.transform(self.y_train_raw)
            self.y_test = self.binarizer.transform(self.y_test_raw)
            self.labels = self.binarizer.classes_
            logging.info('Finished transforming train-test labels into '
                'MultiLabel format')

    def _create_network_graph(self):
        """
        Creates label co-occurence graph as a networkx Graph based on the
        training dataset's labels.

        :return: G - networkx graph of label co-occurences
        """
        from itertools import product
        import networkx as nx

        logging.info("Going to create label network graph with {}"
                     "samples.".format(len(self.y_train)))
        G = nx.Graph()
        for i, row in enumerate(self.y_train):

            if i % 1000 == 0:
                logging.info("Processing row: {}".format(i))

            pos = np.argwhere(row > 0).flatten()
            edge_tuples = list(product(pos, pos))
            for tuple in edge_tuples:
                if G.has_edge(tuple[0], tuple[1]):
                    G[tuple[0]][tuple[1]]['weight'] += 1
                else:
                    G.add_edge(tuple[0], tuple[1], weight=1)

        # logging.info("Finished graph construction.\n  Nodes: {}\n  "
        #              "Edges:{}".format(list(G.nodes), list(G.edges)))
        
        return G

    def _augment_labels(self):

        logging.info("Augmenting labels")
        codes_path = os.path.join(DATA_DIR, "codes.json")
        tree = ICD9(codes_path)

        old_average = np.average([len(row) for row in self.y_raw])

        new_y_raw = []

        for row in self.y_raw:
            new_row = set(row)

            for code in row:

                if len(code) > 3:
                    node = tree.find(code[:3] + "." + code[3:])

                    if node is None:
                        node = tree.find(code[:3] + "." + code[3])
                else:
                    node = tree.find(code)

                if node is not None:
                    for parent in node.parents:
                        new_row.add(parent.code.replace(".", ""))

            new_row.remove("ROOT")
            new_y_raw.append(list(new_row))

        self.y_raw = new_y_raw
        new_average = np.average([len(row) for row in self.y_raw])
        logging.info("Label augmentation complete.\nOld average label "
                     "count per document: {}\nNew average label count per "
                     "document {}".format(old_average, new_average))

    def _top_labels(self, n):
        """
        Filters out labels such that only top n labels in terms of frequency
        remains
        """
        counter = Counter(np.hstack(self.y_raw))
        most_common_labels = set(list(zip(*counter.most_common(n)))[0])
        self.y_raw = [[num for num in arr if num in most_common_labels]
                      for arr in self.y_raw]

    def _create_train_test_dataset(self):
        """
        Creates train and test split for MIMIC-III data. Ignores the labels
        which are not included in the training dataset.
        """
        logging.info("Create train test dataset")
        self.X_train, self.X_test, self.y_train_raw, self.y_test_raw = \
            train_test_split(self.X,
                             self.y_raw,
                             test_size=0.1,
                             random_state=42)

        # Get all unique labels in the training
        train_icd = set(np.unique(np.hstack(self.y_train_raw)))
        logging.info("Training set have {} unique ICD9 codes."
                     .format(len(train_icd)))

        logging.info("Dataset sizes are (examples, labels): \nTrain: ({}, {})"
                     "\nTest: ({}, {})".format(
                         len(self.X_train), len(self.y_train_raw),
                         len(self.X_test), len(self.y_test_raw)))

        logging.info("Remove test dataset labels which are not found in training dataset") 
        # Remove test dataset labels which are not found in training dataset
        self.y_test_raw = [[num for num in arr if num in train_icd]
                           for arr in self.y_test_raw]
        logging.info("Final size of test labels set y_test_raw: {}".format(
            len(self.y_test_raw)))


if __name__ == "__main__":
    data = Dataset(nrows=1000, top_n=100)
