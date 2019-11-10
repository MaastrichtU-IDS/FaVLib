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

import networkx as nx
import random
import numbers
import argparse
import os

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
    #parser.add_argument('-pos', required=True, dest='positive', help="enter postive example file")
    #parser.add_argument('-neg', required=True, dest='negative', help="enter negative exmaple file")
    parser.add_argument('-emb', required=True, dest='embeddings', help="enter embedding file")
    parser.add_argument('-relmap', required=True, dest='relmapping',help="enter folder that contains relation mapping file (relation2id.txt)")
    parser.add_argument('-test', required=True, dest='test', help="enter test fact file")
    parser.add_argument('-otest', required=True, dest='train_output', help="enter file name for prediction output")
    parser.add_argument('-otrain', required=True, dest='test_output', help="enter file name for prediction output")
    
    args = parser.parse_args()
    
    #train_pos_file = args.positive
    #train_neg_file = args.negative
    train_file = args.train
    emb_file = args.embeddings
    relmap_folder = args.relmapping
    test_file = args.test
    train_output = args.train_output
    test_output = args.test_output

    print ("Training file",train_file)
    train_df =pd.read_csv(train_file, names=['Entity1','Relation','Entity2','Class'], sep='\t', header=None)

    emb_df = pd.read_csv(emb_file, delimiter='\t') 

    orgs = set(train_df.Entity1.unique())
    orgs = orgs.union(train_df.Entity2.unique())
    orgs = orgs.intersection(emb_df.Entity.unique())

    print (len(orgs))

    train_df = train_df.merge(emb_df, left_on='Entity1', right_on='Entity').merge(emb_df, left_on='Entity2', right_on='Entity')

    train_df.drop(columns=['Entity_x','Entity_y'], inplace=True)
    
    mapping = {}
    relmap_file  = os.path.join(relmap_folder, 'relation2id.txt')
    with open(relmap_file, 'r') as rmap:
        next(rmap)
        for line in rmap:
            line = line.split('\t')
            mapping[line[0]]= int(line[1])
    

    train_df.Relation = train_df.Relation.replace(mapping)
    print (train_df.head())

    features_cols= train_df.columns.difference(['Entity1','Entity2' ,'Class'])
    X=train_df[features_cols].values
    y=train_df['Class'].values.ravel()



    test_df =pd.read_csv(test_file, names=['Entity1','Relation','Entity2','Class'], sep='\t', header=None)
    
    test_df = test_df.merge(emb_df, left_on='Entity1', right_on='Entity').merge(emb_df, left_on='Entity2', right_on='Entity')

    test_df.drop(columns=['Entity_x','Entity_y'],inplace=True)
    
    print (test_df.head())
    print ("number of positives in test",(len(test_df[test_df['Class']==1])))
    print ("number of neegatives in test",(len(test_df[test_df['Class']!=1])))

    test_df.Relation = test_df.Relation.replace(mapping)
    rf_model = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=10)
    rf_model.fit(X,y)
    X_new=test_df[features_cols].values
    
    rf_scores = get_scores(rf_model, X_new, test_df['Class'])
    print (rf_scores)
   
    id2relation = { value:key for key,value in mapping.items()} 

    test_df.Relation = test_df.Relation.replace(id2relation)
    
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    #probs = rf_model.predict_proba(X_new)
    #test_df['TruthValue'] =  probs[:,1]
    #test_df[['Entity1','Relation','Entity2','TruthValue']].to_csv(output_file, index=False)

