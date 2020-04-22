from sklearn.preprocessing import StandardScaler
from plot_job import plot_cm, plot_aucprc
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
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


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

def fit_model(model, X_train, y_train, X_test, y_test):
    """
    This function train model with X_train, and evaluate with X_test.
    """
    # train model
    model.fit(X_train, y_train)

    # Plot Confusion Matrix and AUC
    plot_cm(model, X_test, y_test)
    plot_aucprc(model, X_test, y_test)

    return model