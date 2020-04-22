from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import fasttext


def victorize_ft(X_train, X_test, col, path='./model/', training=False, dim=300, epoch=25, lr=0.05):
    """
    Using fasttext to vectorize the specified column in X_train and X_test, and replace the original column.
    :param X_train: dataframe
    :param X_test: dataframe
    :param col: string, the name of text column in dataset
    :param path: path to save model
    :param training: Boolean, True to train a new ft model using follwing parameters
    :param dim: int, dimension of vector
    :param epoch: int, the epoches to train model
    :param lr: float, learning rate
    :return: dataframes, replacing original col by vector
    """
    # train ftmodel
    if training:
        X_train[[col]].to_csv(path + "ft.txt", index=False)
        ftmodel = fasttext.train_unsupervised(path + "ft.txt", dim=dim, epoch=epoch, lr=lr)
        ftmodel.save_model(path + "ft.model")
    ftmodel = fasttext.load_model(path + "ft.model")

    # replace text column by ftvector
    X_train_vec = corpus_to_ftvector(ftmodel, X_train, col, dim)
    X_test_vec = corpus_to_ftvector(ftmodel, X_test, col, dim)
    return X_train_vec, X_test_vec


def victorize_cbow(X_train, X_test, col):
    """
    Using CountVectorizer to vectorize the specified column in X_train and X_test, and replace the original column.
    :param X_train: dataframe
    :param X_test: dataframe
    :return: dataframes, replacing original col by vector
    """
    # train CountVectorizer
    count_vectorizer = CountVectorizer()
    X_train_wvec = count_vectorizer.fit_transform(X_train[col])
    X_test_wvec = count_vectorizer.transform(X_test[col])

    # replace the original column
    X_train_wvec = hstack((X_train_wvec, X_train.drop(col, 1).astype(float)))
    X_test_wvec = hstack((X_test_wvec, X_test.drop(col, 1).astype(float)))
    return X_train_wvec, X_test_wvec


def victorize_tfidf(X_train, X_test, col):
    """
    Using TfidfVectorizer to vectorize the specified column in X_train and X_test, and replace the original column.
    :param X_train: dataframe
    :param X_test: dataframe
    :param col: string, the name of text column in dataset
    :return: dataframes, replacing original col by vector
    """
    # train vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    X_train_wvec = tfidf_vectorizer.fit_transform(X_train[col])
    X_test_wvec = tfidf_vectorizer.transform(X_test[col])

    # replace the original column
    X_train_wvec = hstack((X_train_wvec, X_train.drop(col, 1).astype(float)))
    X_test_wvec = hstack((X_test_wvec, X_test.drop(col, 1).astype(float)))
    return X_train_wvec, X_test_wvec


def victorize_glove(X_train, X_test, col, path='../../glove/glove.6B.300d.txt', dim=300):
    """
    Using fasttext to vectorize the specified column in X_train and X_test, and replace the original column.
    :param X_train: dataframe
    :param X_test: dataframe
    :param col: string, the name of text column in dataset
    :param path: path to save model
    :param training: Boolean, True to train a new ft model using follwing parameters
    :param dim: int, dimension of vector
    :param epoch: int, the epoches to train model
    :param lr: float, learning rate
    :return: dataframes, replacing original col by vector
    """
    embeddings_index = {}
    with open(path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], 'float32')
            embeddings_index[word] = vectors
    f.close()

    # use dictionary to convert corpus
    X_train_vec = corpus_to_glvector(embeddings_index, X_train, col, dim)
    X_test_vec = corpus_to_glvector(embeddings_index, X_test, col, dim)
    return X_train_vec, X_test_vec

def corpus_to_ftvector(ftmodel, X_train, col, dim):
    """ use ft model to replace col in X_train
    """
    vec_corp = np.empty((0, dim))
    for text in X_train[col]:
        vec_corp = np.vstack((vec_corp, ftmodel.get_sentence_vector(text)))
    vec = hstack((vec_corp, X_train.drop(col, 1).astype(float)))
    return vec


def corpus_to_glvector(embeddings_index, X, col, dim):
    """ use Glvoe model to replace col in X_train
    """
    X_vec = np.empty((0, dim))
    for text in X[col]:
        X_vec = np.vstack((X_vec, process_words(text, embeddings_index, dim)))
    X_vec = hstack((X_vec, X.drop(col, 1).astype(float)))
    return X_vec

def process_words(text, embeddings_index, dim):
    """ embeddings_index to encode text
    """
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