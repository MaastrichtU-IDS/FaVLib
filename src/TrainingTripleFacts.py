#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
import sys
import pandas as pd
import itertools
import math
import time

from sklearn import svm, linear_model, neighbors
from sklearn import tree, ensemble
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold

import random
import numbers
import argparse
import os
import json

def predict(X_new,  model):

    probs= model.predict_proba(X_new)

    scores = list(zip(test_df['Ent'],test_df['Customer'],probs[:,1]))
    scores.sort(key=lambda tup: tup[2],reverse=True)
    scores_df = pd.DataFrame(scores,columns=['Supplier','Customer','Prob'])

    return scores_df

def get_scores(clf, X_new, y_new):

    scoring = ['precision', 'recall', 'accuracy', 'roc_auc', 'f1', 'average_precision']
    scorers, multimetric = metrics.scorer._check_multimetric_scoring(clf, scoring=scoring)
    #print(scorers)
    scores = multimetric_score(clf, X_new, y_new, scorers)
    return scores


def crossvalid(train_df, test_df): 
    features_cols= train_df.columns.difference(['Entity1','Entity1' ,'Class'])
    X=train_df[features_cols].values
    y=train_df['Class'].values.ravel()

    X_new=test_df[features_cols].values
    y_new=test_df['Class'].values.ravel()

    nb_model = GaussianNB()
    nb_model.fit(X,y)
    nb_scores = get_scores(nb_model, X_new, y_new)

    logistic_model = linear_model.LogisticRegression(C=0.01)
    logistic_model.fit(X,y)
    lr_scores = get_scores(logistic_model, X_new, y_new)

    rf_model = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=10)
    rf_model.fit(X,y)
    rf_scores = get_scores(rf_model, X_new, y_new)

    #sclf_scores = stacking(train_df, test_df)

    #clf = ensemble.RandomForestClassifier(n_estimators=100
    return nb_scores,lr_scores, rf_scores#, sclf_scores

def multimetric_score(estimator, X_test, y_test, scorers):
    """Return a dict of score for multimetric scoring"""
    scores = {}
    for name, scorer in scorers.items():
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%s)"
                             % (str(score), type(score), name))
    return scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-train', required=True, dest='train', help="enter train examples file")
    parser.add_argument('-test', required=True, dest='test', help="enter test fact file")
    #parser.add_argument('-pos', required=True, dest='positive', help="enter postive example file")
    #parser.add_argument('-neg', required=True, dest='negative', help="enter negative exmaple file")
    parser.add_argument('-emb', required=True, dest='embeddings', help="enter embedding file")
    parser.add_argument('-relmap', required=True, dest='relmapping',help="enter mapping file (relation_to_id.json)")
    parser.add_argument('-otrain', required=True, dest='train_output', type=int, help="create file name for train features")
    parser.add_argument('-otest', required=True, dest='test_output', type=int, help="create file name for test features")
    parser.add_argument('-predict', required=True, dest='predict', type=int, help="create file name for prediction output")
    
    args = parser.parse_args()
    
    #train_pos_file = args.positive
    #train_neg_file = args.negative
    train_file = args.train
    emb_file = args.embeddings
    relmap_file = args.relmapping
    test_file = args.test
    train_output = args.train_output
    test_output = args.test_output
    predict  = args.predict
    print(args)

    print ("Training file",train_file)
    train_df =pd.read_csv(train_file, names=['Entity1','Relation','Entity2','Class'], sep='\t', header=None)
    print ("number of train samples",len(train_df))

    emb_df = pd.read_json(emb_file,orient='index')
    emb_df.index.rename('Entity', inplace=True) 


    train_df = train_df.merge(emb_df, left_on='Entity1', right_on='Entity').merge(emb_df, left_on='Entity2', right_on='Entity')

    print ("number of positives in train",(len(train_df[train_df['Class']==1])))
    print ("number of neegatives in train",(len(train_df[train_df['Class']!=1])))
    

    mapping = {}
    with open(relmap_file, 'r') as json_data:
        mapping = (json.load(json_data))
    

    train_df.Relation = train_df.Relation.replace(mapping)
    #print (train_df.head())

    features_cols= train_df.columns.difference(['Entity1','Entity2' ,'Class'])
    X=train_df[features_cols].values
    y=train_df['Class'].values.ravel()



    test_df =pd.read_csv(test_file, names=['Entity1','Relation','Entity2','Class'], sep='\t', header=None)
    print ("number of test samples",len(test_df))
    
    test_df = test_df.merge(emb_df, left_on='Entity1', right_on='Entity').merge(emb_df, left_on='Entity2', right_on='Entity')

    
    #print (test_df.head())
    print ("number of positives in test",(len(test_df[test_df['Class']==1])))
    print ("number of neegatives in test",(len(test_df[test_df['Class']!=1])))

    test_df.Relation = test_df.Relation.replace(mapping)
    #rf_model = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=10)
    #rf_model.fit(X,y)
    X_new=test_df[features_cols].values
    
    #rf_scores = get_scores(rf_model, X_new, test_df['Class'])
    #print (rf_scores)
   
    id2relation = { value:key for key,value in mapping.items()} 

    test_df.Relation = test_df.Relation.replace(id2relation)
    
    folder_out= 'clsout'
    os.mkdir(folder_out)
    if train_output:
        train_df.to_csv(folder_out+'/train_output.csv', index=False)
    if test_output:
        test_df.to_csv(folder_out+'/test_output.csv', index=False)

    results = pd.DataFrame()

    nb_model = GaussianNB()
    lr_model = linear_model.LogisticRegression()
    rf_model = ensemble.RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1)

    clfs = [('Naive Bayes',nb_model),('Logistic Regression',lr_model),('Random Forest',rf_model)]
    bestClf = clfs[0][1]
    bestScore = 0.0
    for name, clf in clfs:
            clf.fit(X, y)
            scores = get_scores(clf, X_new, test_df['Class'])
            scores['method'] = name
            if scores['roc_auc'] > bestScore:
                bestScore= scores['roc_auc'] 
                bestClf = clf
            results = results.append(scores, ignore_index=True)
    
    print ("Best ROC ", bestScore, '\n',bestClf)

    if predict:
        bestClf.fit(X, y)
        test_df['Confidence'] = bestClf.predict_proba(X_new)[:, 1]
        test_df= test_df[test_df['Class']!=1]
        test_df =test_df[['Entity1','Relation','Entity2','Confidence']].sort_values(by='Confidence',ascending=False)
        test_df.to_csv(folder_out+'/prediction.csv', index=False)

    print (results)
    results.to_csv(folder_out+'/results.csv',index=False)
    #probs = rf_model.predict_proba(X_new)
    #test_df['TruthValue'] =  probs[:,1]
    #test_df[['Entity1','Relation','Entity2','TruthValue']].to_csv(output_file, index=False)

