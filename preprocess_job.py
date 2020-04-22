import re
import string
import pandas as pd
import spacy
from gensim.utils import simple_preprocess
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from tqdm import tqdm


def reset_ind(*argv):
    """
    reset the index of dataframe
    :param argv: dataframe
    :return: list of dataframes in sam order
    """
    re = []
    for arg in argv:
        arg = arg.reset_index().drop('index', 1)
        re.append(arg)
    return re


def add_onehot(df, cols, ohe):
    """
    convert categorical cols of df to onehot representation and delete original cols
    :param df: dataframe having specified columns
    :param cols: categorical columns in df
    :param ohe: the trained onehot encoder
    :return: new dataframe with onehot columns
    """
    v = ohe.transform(df[cols]).toarray()
    columns = ohe.get_feature_names()
    df_cate_onehot = pd.DataFrame(v, columns=columns)
    df = df.drop(cols, 1).join(df_cate_onehot)
    return df


def cate_to_onehot(X_train, X_valid, X_test, cate_columns):
    """
    convert categorical cols of df to onehot representation and delete original cols
    :param X_train: Dataframe for training
    :param X_valid: Dataframe for validation
    :param X_test: Dataframe for testing
    :param cate_columns: list of string, categorical columns in dataset
    :return: X_train, X_valid, X_test, with onehot encoded categories
    """
    # Train onehot encoder using training dataset
    ohe_train = OneHotEncoder(
        handle_unknown='ignore')  # ignore tells the encoder to ignore new categories by encoding them with 0's
    ohe_train.fit(X_train[cate_columns])

    # Use encoded onehot to replace original categories
    X_train = add_onehot(X_train, cate_columns, ohe_train)
    X_valid = add_onehot(X_valid, cate_columns, ohe_train)
    X_test = add_onehot(X_test, cate_columns, ohe_train)
    return X_train, X_valid, X_test


def oversampling_smote(X_train, y_train, random_state=42, n_jobs=-1, k_neighbors=5):
    """
    Using SMOTE to oversampling Xtrain and ytrain
    :param X_train: Dataframe
    :param y_train: Dataframe
    :return: new sampled X_train, y_train
    """
    smote = SMOTE(random_state=random_state, n_jobs=n_jobs, k_neighbors=k_neighbors)
    X_train_res, y_train_res = smote.fit_sample(X_train, y_train)
    return X_train_res, y_train_res


def tokenizer_sp(docs):
    """
    tokenize docs use simple_process in Gensim
    :param docs: list of String
    :return: iterator
    """
    for doc in tqdm(docs):
        # set deacc=True to remove wired signs
        yield simple_preprocess(str(doc), deacc=True)


def tokenizer_wt(docs):
    """
    tokenize docs use word_tokenize
    :param docs: list of String
    :return: iterator
    """
    for doc in tqdm(docs):
        yield [reg_filter(w) for w in word_tokenize(str(doc))]


def reg_filter(sentence):
    """
    use regular expression to clean sentence
    :param sentence: string
    :return: string
    """
    text = sentence.lower()
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', '<URL>', text)
    text = re.sub('<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub('\\xa0', ' ', text)
    return text


def remove_stopwords(texts):
    """
    remove stopwords in text
    :param texts: string
    :return: iterator
    """
    stop_words = stopwords.words('english')
    for sentence in tqdm(texts):
        yield [word for word in sentence if word not in stop_words]


def lemmatization(docs, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    lemma docs use spacy
    :param docs: list of String
    :param allowed_postags: list of PoS tags, like 'NOUN', 'ADJ', 'VERB', 'ADV'
    :return: iterator
    """
    texts_out = []
    sp = spacy.load("en_core_web_sm")
    for doc in tqdm(docs):
        s = sp(" ".join(doc))
        text = [token.lemma_ for token in s if token.pos_ in allowed_postags]
        texts_out.append(' '.join(text))
    return texts_out
