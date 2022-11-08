from collections import Counter
import pandas as pd
import pickle
import numpy as np
import logging
from settings import Keys, Tables, DATA_DIR, DATA_COMPRESSION_GZ


class Dataset:
    """
    Creates a dataset for MIMIC-III database discharge summaries with
    corresponding diagnosis codes.

    """

    def __init__(self):
        print('hello world!')
