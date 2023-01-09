"""
    Use the vocabulary to load a matrix of pre-trained word vectors
"""
import numpy as np
import gensim.models
from tqdm import tqdm
from icd_classifier.settings import PAD_CHAR
import logging


def build_matrix(ind2w, wv):
    """
        Go through vocab in order. Find vocab word in wv.index_to_key,
        then call wv.word_vec(wv.index_to_key[i]).
        Put results into one big matrix.
        Note: ind2w starts at 1 (saving 0 for the pad character),
        but gensim word vectors starts at 0
    """
    W = np.zeros(
        (
            len(ind2w) + 1,
            len(wv.word_vec(wv.index_to_key[0]))
        )
    )

    words = [PAD_CHAR]

    W[0][:] = np.zeros(
        len(wv.word_vec(wv.index_to_key[0]))
    )

    for idx, word in tqdm(ind2w.items()):
        if idx >= W.shape[0]:
            break
        W[idx][:] = wv.word_vec(word)
        words.append(word)
    logging.info(
        "Built w2v matrix of shape: {}, from {} words".format(
            W.size(), len(words)))
    return W, words


def gensim_to_embeddings(wv_file, vocab_file, outfile=None):
    """
    output (each item of length 100)
    **PAD** 0.0 .. 0.0
    000cc -0.011251944117248058 .. -0.006207717582583427
    000mcg 0.04674903675913811 .. -0.002568764379248023
    ..
    zytiga -0.04798325151205063 .. -0.023381724953651428
    zyvox -0.0267137810587883 .. -0.0005289725377224386

    output length: 51,918
    """

    model = gensim.models.Word2Vec.load(wv_file)
    wv = model.wv
    # free up memory
    del model

    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i, line in enumerate(vocabfile):
            line = line.strip()
            if line != '':
                vocab.add(line)
    ind2w = {i+1: w for i, w in enumerate(sorted(vocab))}

    W, words = build_matrix(ind2w, wv)

    if outfile is None:
        outfile = wv_file.replace('.w2v', '.embed')
    save_embeddings(W, words, outfile)


def save_embeddings(W, words, outfile):
    with open(outfile, 'w') as o:
        # pad token already included
        for i in range(len(words)):
            line = [words[i]]
            line.extend([str(d) for d in W[i]])
            o.write(" ".join(line) + "\n")
        logging.info(
            "Saved embeddings to file: {}, of shape: {}. "
            "Total words: {} ".format(outfile, W.shape, len(words)))


def load_embeddings(embeddings_file):
    # also normalizes the embeddings
    W = []
    with open(embeddings_file) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
        # UNK embedding, gaussian randomly initialized
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    W = np.array(W)
    logging.info(
        "Load embedding matrix W from embeddings file: {}, matrix shape: {}. "
        "Total matrix size: {} ".format(embeddings_file, W.shape, W.size))
    return W
