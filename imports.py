__author__ = 'osopova'


import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import xgboost as xgb
# from xgboost import XGBClassifier

import datetime

from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 14, 20

from utilities import *

import time

from xgboost.sklearn import XGBClassifier

import json