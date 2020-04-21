from sklearn.preprocessing import StandardScaler
import pandas as pd


def standardize(X_train, X_test, with_mean):
    """
    make
    :param X_train:
    :param X_test:
    :param with_mean:
    :return:
    """
    scaler = StandardScaler(with_mean=with_mean).fit(X_train)
    xtrain_glove_scl = scaler.transform(X_train)
    xtest_glove_scl = scaler.transform(X_test)
    return xtrain_glove_scl, xtest_glove_scl


def load_dataset(path='./data/'):
    """
    read datasets from folder path
    :param path: string, the path to "fake_job_postings.csv" and "country_code.csv"
    :return: two dataframe
    """
    jobPosting = pd.read_csv(path + "fake_job_postings.csv")
    countryCode = pd.read_csv(path + "country_code.csv")
    return jobPosting, countryCode


def split_col(df, col_name):
    """
    to split the col from dataset
    :param df: dataframe of dataset
    :param col_name: string, the name of label
    :return: X, y
    """
    y = df[col_name]
    X = df.drop(col_name, 1)
    return X, y
