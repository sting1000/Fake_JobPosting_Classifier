# Fake Jobposting Detection
## Introduction:
The dataset contains 18K job descriptions out of which about 800 are fake, which is accessable on [Kaggle](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction)

The dataset consists of both textual information and meta-information about the jobs.
And it can be used to create classification models which can learn the job descriptions which are fraudulent.

To have a better understanding of dataset, please download the [data_report](https://github.com/sting1000/Fake_JobPosting_Classifier/blob/master/data_report.html)

## Environment:
* Python 3.6
* Tensorflow 1.15
Other lib please refer to enviroment.yml

## Notebook Outline:
1. Explore Dataset
  * Data overview (using [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling))
  * Process missing value
  * Feature augmentation
2. Feature Selection
  * Distribution for numerical features
  * Bar and accumulation plots for categorical features
3. Text Processing
  * Tokenize 
  * Remove stopwords
  * Lemmatization 
4. Classifiers Related
  * Word Embeddings (CBow, Tf-idf, Fasttext, Glove)
  * Oversampling (SMOTE)
  * Models (Logistic Regression, Random Forest, XGBoost, Neural Network)
4. Bert as encoder (Demo using [bert4keras](https://github.com/bojone/bert4keras))
  * For classifier
  * For Neural Network
