import sys
import os.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier
from stdout_with_time import StdOutWithTime

#sys.stdout = StdOutWithTime(sys.stdout)

n_seed=2016


categories=['ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT','DRIVING UNDER THE INFLUENCE',
            'DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD',
            'GAMBLING','KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL','OTHER OFFENSES',
            'PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES','SEX OFFENSES FORCIBLE',
            'SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS',
            'WEAPON LAWS']


def apply_date_features_transform(df):
    df=df.copy()
    df['Year'] = df.apply(lambda r: int(str(r['Dates'])[0:4]), axis=1)
    df['Month'] = df.apply(lambda r: int(str(r['Dates'])[5:7]), axis=1)
    df['Day'] = df.apply(lambda r: int(str(r['Dates'])[8:10]), axis=1)
    df['Hour'] = df.apply(lambda r: int(str(r['Dates'])[11:13]), axis=1)
    return df.drop(['Dates'], axis=1)


def one_hot_encode_dow(df):
    df=df.copy()
    df_dummy = pd.get_dummies(df['DayOfWeek'], prefix='dow')
    df=df.drop(['DayOfWeek'], axis=1)
    df=pd.concat((df, df_dummy), axis=1)
    return df


def one_hot_encode_dist(df):
    df=df.copy()
    df_dummy = pd.get_dummies(df['PdDistrict'], prefix='d')
    df=df.drop(['PdDistrict'], axis=1)
    df=pd.concat((df, df_dummy), axis=1)
    return df


def take_train_slice(train_x, train_y, limit=100000):
    perm = np.random.permutation(train_x.shape[0])
    train_slice_x = train_x[perm,:]
    train_slice_y = train_y[perm]
    train_slice_x=train_slice_x[:limit,:]
    train_slice_y=train_slice_y[:limit]
    return train_slice_x, train_slice_y


def fix_ll(df):
    df=df.copy()
    delta = 0.5
    x_mean = df['X'].mean()
    df['X']= df.apply(lambda r: x_mean if (r['X'] > x_mean + delta) or  (r['X'] < x_mean - delta) else r['X'], axis=1)

    y_mean = df['Y'].mean()
    df['Y']= df.apply(lambda r: y_mean if (r['Y'] > y_mean + delta) or  (r['Y'] < y_mean - delta) else r['Y'], axis=1)
    return df


def do_cv(x_cv, y_cv, classifier, n_fold):
    perm = np.random.permutation(x_cv.shape[0])
    x_cv = x_cv[perm,:]
    y_cv = y_cv[perm]
    cv_scores = cross_val_score(
        classifier, x_cv, y_cv,
        scoring='log_loss',
        cv=n_fold, verbose=10)
    print('cv_scores: ', cv_scores, '; mean: ', np.mean(cv_scores))


def do_predict(train_x, train_y, test_x, classifier):
    print('training classifier...')
    classifier.fit(train_x, train_y)

    print('predicting...')
    predicted_y = classifier.predict_proba(test_x)

    predicted = pd.DataFrame(predicted_y, columns=categories)

    print('saving submission...')
    predicted.to_csv("submission_xgb_8_500.csv", index_label="Id", na_rep="0", float_format='%.5f')


def do_grid_search(x_search, y_search, classifier, param_grid):
    search_classifier = GridSearchCV(
        clone(classifier),
        param_grid,
        cv=4,
        verbose=10,
        n_jobs=1,
        scoring='log_loss'
    )
    perm = np.random.permutation(x_search.shape[0])
    x_search = x_search[perm,:]
    y_search = y_search[perm]
    search_classifier.fit(x_search, y_search)
    print('grid_scores_: ', search_classifier.grid_scores_)
    print('best_score_: ', search_classifier.best_score_)
    print('best_params_: ', search_classifier.best_params_)
    return search_classifier.best_estimator_


le_cat = LabelEncoder()
le_cat.fit(categories)

print('reading data...')
test = pd.read_csv('data/test.csv')
test=test.drop(['Id', 'Address'], axis=1)

train = pd.read_csv('data/train.csv')
train=train.drop(['Descript', 'Resolution', 'Address'], axis=1)

train_y=le_cat.transform(train['Category'])
train=train.drop(['Category'], axis=1)

if os.path.isfile('train_processed.csv') and os.path.isfile('test_processed.csv'):
    print('reading preprocessed...')
    train = pd.read_csv('train_processed.csv')
    test = pd.read_csv('test_processed.csv')
else:
    print('applying date feature transforming...')
    train = apply_date_features_transform(train)
    test = apply_date_features_transform(test)

    print('applying one hot day of week transform...')
    train=one_hot_encode_dow(train)
    test=one_hot_encode_dow(test)

    print('applying one hot district transform (train)...')
    train=one_hot_encode_dist(train)
    test=one_hot_encode_dist(test)

    print('fixing long & lat...')
    train=fix_ll(train)
    test=fix_ll(test)

    print('saving preprocessed...')
    train.to_csv('train_processed.csv')
    test.to_csv('test_processed.csv')

train_x = train.as_matrix().astype(float)
test_x = test.as_matrix().astype(float)


xgb = XGBClassifier(objective='multi:softprob', max_depth=8, n_estimators=500, nthread=16, seed=n_seed)

# print('grid search xgb...')
# best = do_grid_search(
#     train_x, train_y,
#     XGBClassifier(objective='multi:softprob', max_depth=4, nthread=16, seed=n_seed),
#     {
#         'max_depth':    [5, 6],
#         'n_estimators': [200, 300],
#     })

# print('cross validating...')
# do_cv(train_x, train_y, clone(xgb), 4)

print('predicting...')
do_predict(train_x, train_y, test_x, clone(xgb))

