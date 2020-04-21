from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import fasttext


def victorize_ft(X_train, X_test, col, path='./model/', training=True, dim=300, epoch=25, lr=0.05):
    # train ftmodel
    if training:
        X_train[[col]].to_csv(path + "ft.txt", index=False)
        ftmodel = fasttext.train_unsupervised(path + "ft.txt", dim=dim, epoch=epoch, lr=lr)
        ftmodel.save_model(path + "ft.model")
    ftmodel = fasttext.load_model(path + "ft.model")

    X_train_vec = corpus_to_ftvector(ftmodel, X_train, col, dim)
    X_test_vec = corpus_to_ftvector(ftmodel, X_test, col, dim)
    return X_train_vec, X_test_vec


def victorize_cbow(X_train, X_test, col):
    # train 
    count_vectorizer = CountVectorizer()
    X_train_wvec = count_vectorizer.fit_transform(X_train[col])
    X_test_wvec = count_vectorizer.transform(X_test[col])

    X_train_wvec = hstack((X_train_wvec, X_train.drop(col, 1).astype(float)))
    X_test_wvec = hstack((X_test_wvec, X_test.drop(col, 1).astype(float)))
    return X_train_wvec, X_test_wvec


def load_glove(path='../../glove/glove.6B.200d.txt', dim=200):
    # Glove
    embeddings_index = {}
    with open(path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], 'float32')
            embeddings_index[word] = vectors
    f.close()
    return embeddings_index


def victorize_glove(X_train, X_test, col, embedding, dim=200):
    X_train_vec = np.empty((0, dim))
    for text in X_train[col]:
        X_train_vec = np.vstack((X_train_vec, process_words(text, embedding, dim)))
    X_train_vec = hstack((X_train_vec, X_train.drop(col, 1).astype(float)))

    X_test_vec = np.empty((0, dim))
    for text in X_test[col]:
        X_test_vec = np.vstack((X_test_vec, process_words(text, embedding, dim)))
    X_test_vec = hstack((X_test_vec, X_test.drop(col, 1).astype(float)))

    return X_train_vec, X_test_vec


def process_words(text, embeddings_index, dim):
    M = []
    for w in text:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(dim)
    return v / np.sqrt((v ** 2).sum())


def corpus_to_ftvector(ftmodel, X_train, col, dim):
    vec_corp = np.empty((0, dim))
    for text in X_train[col]:
        vec_corp = np.vstack((vec_corp, ftmodel.get_sentence_vector(text)))
    vec = hstack((vec_corp, X_train.drop(col, 1).astype(float)))
    return vec
