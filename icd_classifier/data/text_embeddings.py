"""
    Pre-train embeddings using gensim w2v implementation (CBOW by default)
"""
import csv
import logging
import numpy as np


class ProcessedIter(object):

    def __init__(self, Y, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                yield (row[3].split())


def word2vec_embeddings(Y, notes_file, embedding_size, min_count, epochs):
    import gensim.models.word2vec as w2v

    modelname = "processed_%s.w2v" % (Y)
    sentences = ProcessedIter(Y, notes_file)

    model = w2v.Word2Vec(
        vector_size=embedding_size, min_count=min_count, workers=4)
    logging.debug("Building word2vec vocab on %s..." % (notes_file))
    model.build_vocab(sentences)

    logging.debug("Training word2vec")
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    out_file = '/'.join(notes_file.split('/')[:-1] + [modelname])

    logging.debug("Writing word2vec embeddings to %s" % (out_file))
    model.save(out_file)
    return out_file


def sentence_embeddings(model_name, text_file, embeddings_file):
    """
    Usage:
    model_name = 'all-mpnet-base-v2'
    model_name = 'BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
    model_full_name = 'pritamdeka/'+model_name
    text_file = 'data/processed/Z.all.full.txt'
    embeddings_file = 'data/processed/Z.emb.all.'+model_name+'.npy'

    embed_text_file(model_full_name, text_file, embeddings_file)
    """

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    with open(text_file) as f:
        sentences = f.readlines()

    sentences = [sentence.rstrip('\n') for sentence in sentences]
    print(len(sentences))

    embeddings = model.encode(sentences)

    print(embeddings.shape)
    embeddings = model.encode(sentences)

    np.save(embeddings_file, embeddings)
