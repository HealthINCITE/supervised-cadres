#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed June 05 17:07 2019

@authors: jkuruzovich, karhin
"""
import sys 

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.datasets.samples_generator import make_blobs
from  sklearn.datasets import make_classification
from  sklearn.datasets import make_regression
from matplotlib import pyplot
import pandas as pd    
import numpy as np

class DGenerator:
    def generate_cadres(samples, features, informative, cadres, seed):
        """
        samples = sample size
        features = cadre features 
        cadres = number of cadres
        informative = random state
        seed = random state
        """
        samples_per_cadre=int(samples/cadres)
        X, y= make_classification(n_samples=samples, n_features=features, n_informative=informative,  n_classes=cadres,  random_state=seed)
        c_columns = ['cad'+str(x) for x in range (features)] #['cad'+str(x) for x in range (cf)]
        df = pd.DataFrame(X, columns = c_columns)
        df["cadre"] = y
        df=df.sort_values(by=['cadre'],)
        df=df.reset_index(drop=True)
        df['index_c']= [x for y in range(cadres) for x in range(samples_per_cadre)]
        return df

    def generate_variables(df, samples, features, classes, informative, seed):
        """        
        df = dataframe from generate_cadre function
        samples = sample size
        features = depedent variables
        classes = target / dependent variable / number of responses for categorical variable
        informative = number of informative features in features
        seed = random state
        """
        df2=pd.DataFrame()
        cadres = np.array(df['cadre'])
        
        for cadre in range(cadres):
            seed_c=cadre*seed
            X, y= make_classification(n_samples=samples, n_features=features, n_informative=informative,  n_classes=classes,  random_state=seed_c)
            columns= [ 'dv'+str(x) for x in range(features)]
            df3 = pd.DataFrame(X, columns = columns)
            df3["target"]=y
            df3["cadre"]=cadre
            df3["index_c"]=df3.index
            df2=df2.append(df3)

        return df2
