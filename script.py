import os.path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.cross_validation import cross_val_score
from xgboost.sklearn import XGBClassifier

n_seed = 2016
n_threads = 16


categories=['ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT','DRIVING UNDER THE INFLUENCE',
            'DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD',
            'GAMBLING','KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL','OTHER OFFENSES',
            'PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES','SEX OFFENSES FORCIBLE',
            'SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS',
            'WEAPON LAWS']


def apply_date_features_transform(df):
    df = df.copy()
    df['Year'] = df.apply(lambda r: int(str(r['Dates'])[0:4]), axis=1)
    df['Month'] = df.apply(lambda r: int(str(r['Dates'])[5:7]), axis=1)
    df['Day'] = df.apply(lambda r: int(str(r['Dates'])[8:10]), axis=1)
    df['Min_of_day'] = df.apply(lambda r: 60*int(str(r['Dates'])[11:13]) + int(str(r['Dates'])[14:16]), axis=1)
    return df.drop(['Dates'], axis=1)


def one_hot_encode_dow(df):
    df = df.copy()
    df_dummy = pd.get_dummies(df['DayOfWeek'], prefix='dow')
    df = df.drop(['DayOfWeek'], axis=1)
    df = pd.concat((df, df_dummy), axis=1)
    return df


def one_hot_encode_dist(df):
    df = df.copy()
    df_dummy = pd.get_dummies(df['PdDistrict'], prefix='d')
    df = df.drop(['PdDistrict'], axis=1)
    df = pd.concat((df, df_dummy), axis=1)
    return df


def fix_ll(df):
    df = df.copy()
    delta = 0.5
    x_mean = df['X'].mean()
    df['X']= df.apply(lambda r: x_mean if (r['X'] > x_mean + delta) or  (r['X'] < x_mean - delta) else r['X'], axis=1)

    y_mean = df['Y'].mean()
    df['Y']= df.apply(lambda r: y_mean if (r['Y'] > y_mean + delta) or  (r['Y'] < y_mean - delta) else r['Y'], axis=1)
    return df


def get_frequent_addresses(df, count):
    return list(df['Address'].value_counts().head(count).to_dict().keys())


def add_address_as_features(df, addresses):
    df = df.copy()
    for i, address in enumerate(addresses):
        df['address_' + str(i)] = df.apply(lambda r: 1 if r['Address'] == address else 0, axis=1)
    return df.drop(['Address'], axis=1)


def do_cv(x_cv, y_cv, classifier, n_fold):
    perm = np.random.permutation(x_cv.shape[0])
    x_cv = x_cv[perm,:]
    y_cv = y_cv[perm]
    cv_scores = cross_val_score(
        classifier, x_cv, y_cv,
        scoring='log_loss',
        cv=n_fold, verbose=10)
    print('cv_scores: ', cv_scores, '; mean: ', np.mean(cv_scores))


def do_predict(x_train_predict, y_train_predict, x_test_predict, classifier, submission_path):
    print('training classifier...')
    classifier.fit(x_train_predict, y_train_predict)

    print('predicting...')
    predicted_y = classifier.predict_proba(x_test_predict)

    predicted = pd.DataFrame(predicted_y, columns=categories)

    print('saving submission...')
    predicted.to_csv(submission_path, index_label="Id", na_rep="0", float_format='%.5f')


le_cat = LabelEncoder()
le_cat.fit(categories)

print('reading data...')
test = pd.read_csv('data/test.csv')
test = test.drop(['Id'], axis=1)

train = pd.read_csv('data/train.csv')
train = train.drop(['Descript', 'Resolution' ], axis=1)

train_y = le_cat.transform(train['Category'])
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
    train = one_hot_encode_dow(train)
    test = one_hot_encode_dow(test)

    print('applying one hot district transform (train)...')
    train = one_hot_encode_dist(train)
    test = one_hot_encode_dist(test)

    print('fixing long & lat...')
    train = fix_ll(train)
    test = fix_ll(test)

    print('featurizing address...')
    frequent_addresses = get_frequent_addresses(train, 250)
    train = add_address_as_features(train, frequent_addresses)
    test = add_address_as_features(test, frequent_addresses)

    print('saving preprocessed...')
    train.to_csv('train_processed.csv')
    test.to_csv('test_processed.csv')

train_x = train.as_matrix().astype(float)
test_x = test.as_matrix().astype(float)

xgb = XGBClassifier(objective='multi:softprob', max_depth=10, n_estimators=500,
                    nthread=n_threads, seed=n_seed)

print('cross validating...')
do_cv(train_x, train_y, clone(xgb), 4)

print('predicting...')
do_predict(train_x, train_y, test_x, clone(xgb), "submission_250_xg10_500.csv")
