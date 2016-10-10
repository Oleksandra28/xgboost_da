__author__ = 'osopova'

from imports import *
from sklearn.cross_validation import train_test_split, cross_val_score

#######################################################################################
### 1. load data. training and testing data encoded as one-hot
data_folder = './data'

sandyData = np.loadtxt(data_folder + '/sandyData.csv', delimiter=',')
sandyLabels = np.loadtxt(data_folder + '/sandyLabels.csv', delimiter=',')

test_size = 0.95
random_state = 7

X_train, x_test, y_train, y_test = \
    train_test_split(sandyData, sandyLabels, test_size=test_size,
                     random_state=random_state)

#######################################################################################

xgb1 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)

modelfit(xgb1, X_train, y_train)

# Step 2: Tune max_depth and min_child_weight
param_test1 = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}
gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch1.fit(X_train, y_train)
print "gsearch1.grid_scores_"
print gsearch1.grid_scores_
print "gsearch1.best_params_"
print gsearch1.best_params_
print "gsearch1.best_score_"
print gsearch1.best_score_

filename = generate_unique_filename() + '.txt'

# with open(filename,'a+') as f:
# f.write("gsearch1.grid_scores_")
#     np.savetxt(f,gsearch1.grid_scores_)
#     f.write("gsearch1.best_params_")
#     np.savetxt(f,gsearch1.best_params_)
#     f.write("gsearch1.best_score_")
#     np.savetxt(f,gsearch1.best_score_)



# .... Step 2: Tune max_depth and min_child_weight

param_test2 = {
    'max_depth': [8, 9, 10],
    'min_child_weight': [2, 3, 4]
}
gsearch2 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test2, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch2.fit(X_train, y_train)
print "gsearch2.grid_scores_"
print gsearch2.grid_scores_
print "gsearch2.best_params_"
print gsearch2.best_params_
print "gsearch2.best_score_"
print gsearch2.best_score_
print "================================================================================"

# Step 3: Tune gamma
param_test3 = {
    'gamma': [i / 10.0 for i in range(0, 5)]
}
gsearch3 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=4,
                                                min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch3.fit(X_train, y_train)
print "gsearch3.grid_scores_"
print gsearch3.grid_scores_
print "gsearch3.best_params_"
print gsearch3.best_params_
print "gsearch3.best_score_"
print gsearch3.best_score_
print "================================================================================"

# Step 4: Tune subsample and colsample_bytree

param_test4 = {
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)]
}
gsearch4 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=4,
                                                min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test4, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch4.fit(X_train, y_train)
print "gsearch4.grid_scores_"
print gsearch4.grid_scores_
print "gsearch4.best_params_"
print gsearch4.best_params_
print "gsearch4.best_score_"
print gsearch4.best_score_
print "================================================================================"
param_test5 = {
    'subsample': [i / 100.0 for i in range(75, 90, 5)],
    'colsample_bytree': [i / 100.0 for i in range(75, 90, 5)]
}
gsearch5 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=4,
                                                min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test5, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch5.fit(X_train, y_train)
print "gsearch5.grid_scores_"
print gsearch5.grid_scores_
print "gsearch5.best_params_"
print gsearch5.best_params_
print "gsearch5.best_score_"
print gsearch5.best_score_
print "================================================================================"
param_test6 = {
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=4,
                                                min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test6, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch6.fit(X_train, y_train)
print "gsearch6.grid_scores_"
print gsearch5.grid_scores_
print "gsearch6.best_params_"
print gsearch5.best_params_
print "gsearch6.best_score_"
print gsearch5.best_score_
print "================================================================================"
