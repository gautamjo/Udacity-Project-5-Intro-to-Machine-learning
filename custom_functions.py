import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def find_outlier(df_in, col_name):
    '''Returns a data frame of values that exit outside the 
    interquartile range of the data. This function requires two parameter 
    first is a dataframe and second is a column name which is a string.'''
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    #df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    df_i = df_in.loc[(df_in[col_name] < fence_low) | (df_in[col_name] > fence_high)]
    return df_i


def train_predict_evaluate(clf, feats, labls, new_feats, new_labls, iters = 20):
    '''Returns a data frame of metric scores after predictions. 
    '''
    original_score_lst = []
    original_accuracy = []
    original_recall = [] 
    original_precision = []

    new_score_lst = []
    new_accuracy = []
    new_recall = []
    new_precision = []

    for i in range(iters):

        # extracting train and test set without new features
        features_train, features_test, labels_train, labels_test = \
        train_test_split(feats, labls, test_size=0.3, random_state=i)

        # extracting train and test set with new features
        features_train1, features_test1, labels_train1, labels_test1 = \
        train_test_split(new_feats, new_labls, test_size=0.3, random_state=i)

        # classifying and predicting without new features
        original_classifier = clf
        original_classifier.fit(features_train, labels_train)
        original_pred = original_classifier.predict(features_test)

        # classifying and predicting with new features
        new_classifier = clf
        new_classifier.fit(features_train1, labels_train1)
        new_pred = new_classifier.predict(features_test1)

        # model scores without new features
        original_accuracy.append(round(accuracy_score(labels_test, original_pred), 3))
        original_recall.append(round(recall_score(labels_test, original_pred), 3))
        original_precision.append(round(precision_score(labels_test, original_pred), 3))


        # model scores with new features
        new_accuracy.append(round(accuracy_score(labels_test1, new_pred), 3))
        new_recall.append(round(recall_score(labels_test1, new_pred), 3))
        new_precision.append(round(precision_score(labels_test1, new_pred), 3))

    original_score_lst.append(np.mean(original_accuracy))
    original_score_lst.append(np.mean(original_precision)) 
    original_score_lst.append(np.mean(original_recall))

    new_score_lst.append(np.mean(new_accuracy))
    new_score_lst.append(np.mean(new_precision))
    new_score_lst.append(np.mean(new_recall))

    df_score = pd.DataFrame({'Metric' : ['Accuracy', 'Precision', 'Recall'], 
                            'Original_score' : original_score_lst,
                            'New_score' : new_score_lst})
    
    return df_score

def score_chart(df, clf_name):
    '''This function plots a bar chart of model scores.
    '''
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize = (12, 6))

    bar = df.plot(kind = 'bar', x='Metric', y='Original_score', ax=ax1)
    ax1.set_title(clf_name + " scores without new features")
    for p in bar.patches:
            bar.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

    bar1 = df.plot(kind = 'bar', x='Metric', y='New_score', ax=ax2,
                       color = 'green')
    ax2.set_title(clf_name + " score with new features")
    for p in bar1.patches:
            bar1.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

    plt.tight_layout()
    plt.show()