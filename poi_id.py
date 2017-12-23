# importing libraries
from __future__ import division
import pickle
import math
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pprint import pprint
sys.path.append("../tools/")
from custom_functions import find_outlier, train_predict_evaluate, score_chart
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score

# loading data
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
# creating a dataframe from dict
df = pd.DataFrame.from_dict(data_dict).T
df.reset_index(inplace = True)
df.rename(columns = {'index' : 'name'}, inplace = True)

# writing dataframe to .csv
df.to_csv('enron_data.csv')

# reading .csv into a dataframe
enron_df = pd.read_csv('enron_data.csv')
enron_df.drop('Unnamed: 0', axis = 1, inplace = True)


# checking names
print('Names of all enron executive in the dataset: \n{}'.format([name for name in enron_df['name']]))

# total number of names
print('Total number of names in the dataset: {}'.format(len(enron_df['name'])))

# checking feature list
feature_set = list(enron_df.drop('name', axis = 1))
print 'Feature List:\n',feature_set
print
print 'Number of features:', len(feature_set)

# checking for POI names
poi_name = enron_df.loc[enron_df['poi'] == True]['name']
print('Names of POIs:\n{}'.format(poi_name))
print ''
print 'Number of Persons Of Interest:', len(poi_name )
print ''
print 'Number of Non Person of Interset:', len(enron_df['name']) - len(poi_name)


# Task 1: Select what features you'll use.

financial_features = ['salary', 'deferral_payments', 'total_payments',
                     'exercised_stock_options', 'bonus', 'restricted_stock',
                     'restricted_stock_deferred', 'total_stock_value', 'expenses', 
                     'loan_advances', 'director_fees', 'deferred_income', 'long_term_incentive']

email_features = ['to_messages', 'shared_receipt_with_poi', 'from_messages', 'other', 
                 'from_this_person_to_poi', 'email_address', 'from_poi_to_this_person']

poi = ['poi']

total_features = poi + financial_features + email_features

print 'Outlier Names:'
print(enron_df[(enron_df['name'] == 'TOTAL') | (enron_df['name'] == 'THE TRAVEL AGENCY IN THE PARK') 
        | (enron_df['name'] == 'LOCKHART EUGENE E')].T)


# Task 2: Remove outliers from data_dict
data_dict.pop('TOTAL', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
print('Three outliers removed: {}, {}, {}'.format('TOTAL', 'LOCKHART EUGENE E', 'THE TRAVEL AGENCY IN THE PARK'))

# removing outliers from dataframe
enron_df.drop(enron_df.index[[84, 127, 130]], inplace = True)

# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
my_dataset = data_dict


def ratio(feature1, feature2):
    '''Returns a ratio of two features, which are strings, if they aren't 
    equal to NaN. Otherwise returns a 0.
    '''
    
    if feature1 != 'NaN'and feature2 != 'NaN':
        return feature1/feature2
    
    return 0

def new_feature(feature1, feature2, new_feature):
    '''This function updates a dictionary named my_dataset with a new feature. 
    Takes in three strings.
    '''
    for name in my_dataset:
        my_dataset[name][new_feature] = ratio(my_dataset[name][feature1], 
                                              my_dataset[name][feature2])
    
# creating new features called from_ratio
new_feature('from_poi_to_this_person', 'to_messages', 'from_ratio')

# creating a new feature called to_ratio
new_feature('from_this_person_to_poi', 'from_messages', 'to_ratio')

# creating new features called 'exercised_stock_to_salary_ratio'
new_feature('exercised_stock_options', 'salary', 'exercised_stock_to_salary_ratio')

# creating a new feature called bonus_to_salary_ratio
new_feature('bonus', 'salary', 'bonus_to_salary_ratio')

# creating a new feature called incentives_to_salary_ratio
new_feature('long_term_incentive', 'salary', 'incentives_to_salary_ratio')


print'Dataset with new features:\n', my_dataset.values()[0].keys()

# adding new features to the list of features
new_features_list = total_features + ['from_ratio', 'to_ratio', 'exercised_stock_to_salary_ratio',
                                      'bonus_to_salary_ratio', 'incentives_to_salary_ratio']

# removing email_address
new_features_list.remove('email_address')

# extracting features and labels from dataset
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# selecting the top 10 features
selector = SelectKBest(f_classif, k = 10)
selector.fit_transform(features, labels)
scores = zip(new_features_list[1:], selector.scores_)
sorted_scores = sorted(scores, key = lambda x:x[1], reverse = True)
best_features_list = [new_features_list[0]] + list(map(lambda x: x[0], sorted_scores))[0:9]
print('Best selected features:\n{}'.format(sorted_scores[:9]))

# removing new features to retain only the original features
best_features_list.remove('to_ratio')
best_features_list.remove('bonus_to_salary_ratio')

# Extracting and scaling best features
new_data = featureFormat(my_dataset, best_features_list, sort_keys = True)
labels, features = targetFeatureSplit(new_data)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Extracting and scaling new features
updated_data = featureFormat(my_dataset, best_features_list + new_features_list[-5:], 
                             sort_keys = True)
new_labels, new_features = targetFeatureSplit(updated_data)
new_scaled_features = scaler.fit_transform(new_features)


# predicting and plotting GaussianNB
predict = train_predict_evaluate(GaussianNB(), scaled_features, labels, new_scaled_features, new_labels)
score_chart(predict, 'Gaussian Naive Bayes')


# predicting and plotting SVC
predict = train_predict_evaluate(SVC(), scaled_features, labels, new_scaled_features, new_labels)
score_chart(predict, 'Support Vector Classifier')


# predicting and plotting DecisionTreeClassifier
predict = train_predict_evaluate(DecisionTreeClassifier(), scaled_features, labels, new_scaled_features, new_labels)
score_chart(predict, 'Decision Tree Classifier')


# predicting and plotting RandomForrestClassifier
predict = train_predict_evaluate(RandomForestClassifier(), scaled_features, labels, new_scaled_features, new_labels)
score_chart(predict, 'Random Forrest Classifier')


# predicting and plotting LogisticRegression
predict = train_predict_evaluate(LogisticRegression(), scaled_features, labels, new_scaled_features, new_labels)
score_chart(predict, 'Logistic Regression')

def parameter_tuning(grid_search, features, labels, params, iters = 100):
    """ This function tunes the algorithm using grid search and prints out the 
    average evaluation metric results after performing the tuning for iter times,
    and the best hyperparameters for the model.
    """
    accuracy = []
    precision = []
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size = 0.3, random_state = i)
        grid_search.fit(features_train, labels_train)
        predicts = grid_search.predict(features_test)

        accuracy.append(round(accuracy_score(labels_test, predicts),3))
        precision.append(round(precision_score(labels_test, predicts),3))
        recall.append(round(recall_score(labels_test, predicts),3))
    print "accuracy: {}".format(np.mean(accuracy))
    print "precision: {}".format(np.mean(precision))
    print "recall: {}".format(np.mean(recall))
    
    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print 'Best Features:'
        print("%s = %r, " % (param_name, best_params[param_name]))

def model_evaluation(classifier, parameters, algo_name):
    '''This function conducts a grid search and calls the function 
    parameter_tuning to print metric scores and best parameters.'''
    
    clf = classifier
    param = parameters
    grid_search = GridSearchCV(estimator = clf, param_grid = param)
    print(algo_name + " evaluation:")
    parameter_tuning(grid_search, scaled_features, labels, param)
    #print ''
    #parameter_tuning(nb_grid_search, new_scaled_features, new_labels, nb_param)


model_evaluation(GaussianNB(), {}, 'Gaussian Naive Bayes')

# dtc_param = {'criterion':('gini', 'entropy'),
#                   'splitter':('best','random'),
#                   'min_samples_leaf':[1,2,3,4,5], 
#                   'min_samples_split':[2,3,4,5]}

# model_evaluation(DecisionTreeClassifier(), dtc_param, 'Decision Tree Classifier')

# svm_param = {'kernel':('linear', 'rbf', 'sigmoid'),
#              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#              'C': [0.1, 1, 10, 100, 1000]}

# model_evaluation(SVC(), svm_param, 'Support vector Classifier')

# rf_param = {'min_samples_leaf':[1,2,3,4,5], 
#             'min_samples_split':[2,3,4,5], 
#             'n_estimators':[5,10,20,30]}

# model_evaluation(RandomForestClassifier(), rf_param, 'Random Forrest Classifier')

# logR_param =   {"C":[0.05, 0.5, 1, 10, 100, 1000, 10**5, 10**10],
#                     "tol":[0.1, 10**-5, 10**-10]}
# model_evaluation(LogisticRegression(), logR_param, 'Logistic Regression')


clf = GaussianNB()
features_list = best_features_list
test_classifier(clf, my_dataset, features_list)

#Dump your classifier, dataset, and features_list so 
#anyone can run/check your results.
dump_classifier_and_data(clf, my_dataset, features_list)













