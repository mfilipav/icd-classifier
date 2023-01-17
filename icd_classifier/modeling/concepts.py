from scipy.sparse import csr_matrix


def understand_csr_matrix():
    docs = [
        ["hello", "world", "hello"],
        ["goodbye", "cruel", "goodbye", "world"]]
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}

    for d in docs:
        print("doc: ", d)
        for term in d:
            print("  term: ", term)
            index = vocabulary.setdefault(term, len(vocabulary))
            print("  vocabulary: ", vocabulary)
            print("  index: ", index)
            indices.append(index)
            print("  indices: ", indices)
            data.append(1)
            print("  data: ", data)
        indptr.append(len(indices))
        print("indptr: ", indptr, "\n")

    # indices:  [0, 1, 0, 2, 3, 2, 1]
    # data:  [1, 1, 1, 1, 1, 1, 1]
    # indptr:  [0, 3, 7]
    # <2x4 sparse matrix of type '<class 'numpy.int64'>'
    #  with 7 stored elements in Compressed Sparse Row format>
    M = csr_matrix((data, indices, indptr), dtype=int)

    # array([[2, 1, 0, 0],
    #        [0, 1, 2, 1]])
    # equivalent M.toarray()
    M.A

    # Convert this matrix to COOrdinate format
    Mc = M.tocoo()
    # array([0, 0, 0, 1, 1, 1, 1], dtype=int32)
    Mc.row  # mask defined by indptr=[0, 3, 7]

    # array([0, 1, 0, 2, 3, 2, 1], dtype=int32)
    Mc.col  # stores indices
    csr_matrix((data, indices, indptr), dtype=int).toarray()
