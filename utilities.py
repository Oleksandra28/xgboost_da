__author__ = 'osopova'

from imports import *


import time
import pickle

### from https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
def modelfit(alg, train, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, verb_eval=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train, label=target)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=verb_eval)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(train, target, eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(train)
    dtrain_predprob = alg.predict_proba(train)[:,1]

    print "dtrain_predictions"
    print type(dtrain_predictions)
    dtrain_predictions = dtrain_predictions.astype(dtype=int, copy=False)
    print dtrain_predictions

    np.savetxt(generate_unique_filename() + "-predictions.csv", dtrain_predictions, delimiter=",")

    # xgb.save_model(alg, generate_unique_filename()+'.model')
    # # dump model with feature map
    # xgb.dump_model(alg, generate_unique_filename()+'-raw.txt',generate_unique_filename()+'-featmap.txt')

    # save model to file
    pickle.dump(alg, open(generate_unique_filename()+"-model-pickle.dat", "wb"))


    s = "\nModel Report " + generate_unique_filename() +\
        "\nAccuracy : %.4g" % accuracy_score(target, dtrain_predictions) +\
        "\nAUC Score (Train): %f" % roc_auc_score(target, dtrain_predprob) +\
        "\nF1 score: %f" % f1_score(target, dtrain_predictions)

    #Print model report:
    print s

    filename = generate_unique_filename()+'.txt'
    file = open(filename, 'a+')

    file.write(s)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    filename = generate_unique_filename()+'.png'
    plt.savefig(filename)


def generate_unique_filename():
    fmt = "%Y-%m-%d-%H-%M-%S"
    filename = time.strftime(fmt)
    filename = './output/'+filename
    return filename
