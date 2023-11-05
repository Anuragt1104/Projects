import sys, os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from datetime import datetime

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier


def get_features(dataset, cond_vectorizer, review_vectorizer, method='transform'):
    dataset['review'] = dataset['review'].fillna('')
    dataset.dropna(axis=0, inplace=True)

    stop_words = set(stopwords.words('english'))
    dataset['review'] = dataset['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    if method == 'fit':
        X1 = cond_vectorizer.fit_transform(dataset['condition'])
        X2 = review_vectorizer.fit_transform(dataset['review'])
        X = hstack([X1, X2])
    else:
        X1 = cond_vectorizer.transform(dataset['condition'])
        X2 = review_vectorizer.transform(dataset['review'])
        X = hstack([X1, X2])
    
    X_d = [datetime.strptime(date, '%B %d, %Y').date() for date in dataset['date']]
    X_d = [[d.day, d.month, d.year] for d in X_d]
    
    X = hstack([X, X_d, dataset[['usefulCount']].values])
    y = dataset['rating']

    return X, y


def write_accuracies(model, filename, X_train, y_train, X_test, y_test, X_val, y_val):
    with open(filename, 'a+') as f:
        with redirect_stdout(f):
            print('Training Accuracy: ', model.score(X_train, y_train))
            print('Testing Accuracy: ', model.score(X_test, y_test))
            print('Validation Accuracy: ', model.score(X_val, y_val))


def a_decision_tree(X_train, y_train, X_test, y_test, X_val, y_val, filename, subpart='a'):
    dt_model = DecisionTreeClassifier(random_state=0)
    dt_model.fit(X_train, y_train)
    write_accuracies(dt_model, filename, X_train, y_train, X_test, y_test, X_val, y_val)


def b_dt_gridsearch(X_train, y_train, X_test, y_test, X_val, y_val, filename, subpart='b'):
    params = {
        'max_depth': [20, 40, 60],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3],
    }
    cv = np.concatenate((-1*np.ones((X_train.shape[0], 1)), np.zeros((X_val.shape[0], 1))), axis=0)
    cv = PredefinedSplit(cv)
    gs_model = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=params, cv=cv, scoring='accuracy', n_jobs=-1, refit=True, error_score='raise')

    X_cv = vstack([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val], axis=0)

    gs_model.fit(X_cv, y_cv)

    write_accuracies(gs_model, filename, X_train, y_train, X_test, y_test, X_val, y_val)
    with open(filename, 'a') as f:
        f.write(gs_model.best_params_)


def c_ccp(X_train, y_train, X_test, y_test, X_val, y_val, filename, OUTPUT_FOLDER_PATH, subpart='c'):
    ccp_path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = ccp_path.ccp_alphas, ccp_path.impurities

    # print(ccp_alphas.shape)
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle='steps-post')
    ax.set_xlabel('effective alpha')
    ax.set_ylabel('total impurity of leaves')
    ax.set_title('Total Impurity vs effective alpha for training set')
    fig.savefig(os.path.join(OUTPUT_FOLDER_PATH, '2_'+subpart+'_alp_vs_imp.png'), dpi=300)

    ccp_models = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        ccp_models.append(clf)

    ccp_models = ccp_models[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [model.tree_.node_count for model in ccp_models]
    depth = [model.tree_.max_depth for model in ccp_models]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle='steps-post')
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker='o', drawstyle='steps-post')
    ax[1].set_xlabel('alpha')
    ax[1].set_ylabel('depth of tree')
    ax[1].set_title('Depth vs alpha')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER_PATH, '2_'+subpart+'_nodes_vs_alpha_depth_vs_alpha.png'), dpi=300)

    train_scores = [model.score(X_train, y_train) for model in ccp_models]
    test_scores = [model.score(X_test, y_test) for model in ccp_models]
    val_scores = [model.score(X_val, y_val) for model in ccp_models]

    fig, ax = plt.subplots()
    ax.set_xlabel('alpha')
    ax.set_ylabel('accuracy')
    ax.set_title('Accuracy vs alpha for training and testing sets')
    ax.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle='steps-post')
    ax.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle='steps-post')
    ax.plot(ccp_alphas, val_scores, marker='o', label='validation', drawstyle='steps-post')
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_FOLDER_PATH, '2_'+subpart+'_accuracy_vs_alpha.png'), dpi=300)

    ccp_model = ccp_models[np.argmax(val_scores)]
    write_accuracies(ccp_model, filename, X_train, y_train, X_test, y_test, X_val, y_val)


def d_random_forest(X_train, y_train, X_test, y_test, X_val, y_val, filename, subpart='d'):
    params = {
        'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450],
        'max_features': [0.4, 0.5, 0.6, 0.7, 0.8],
        'min_samples_split': [2, 4, 6, 8, 10],
    }

    cv = np.concatenate((-1*np.ones((X_train.shape[0], 1)), np.zeros((X_val.shape[0], 1))), axis=0)
    cv = PredefinedSplit(cv)
    rf_model = GridSearchCV(estimator=RandomForestClassifier(oob_score=True, random_state=0), param_grid=params, cv=cv, scoring='accuracy', n_jobs=-1, refit=True, error_score='raise')
    
    X_cv = vstack([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val], axis=0)
    rf_model.fit(X_cv, y_cv)

    write_accuracies(rf_model, filename, X_train, y_train, X_test, y_test, X_val, y_val)
    with open(filename, 'a') as f:
        with redirect_stdout(f):
            print('OOB Accuracy: ', rf_model.best_estimator_.oob_score_)
            print(rf_model.best_params_)


def e_xgb(X_train, y_train, X_test, y_test, X_val, y_val, filename, subpart='e'):
    params = {
        'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450],
        'subsample': [0.4, 0.5, 0.6, 0.7, 0.8],
        'max_depth': [40, 50, 60, 70],
    }

    cv = np.concatenate((-1*np.ones((X_train.shape[0], 1)), np.zeros((X_val.shape[0], 1))), axis=0)
    cv = PredefinedSplit(cv)
    xgb_model = GridSearchCV(estimator=XGBClassifier(random_state=0, n_jobs=-1), param_grid=params, cv=cv, scoring='accuracy', n_jobs=-1, refit=True, error_score='raise')
    
    X_cv = vstack([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val], axis=0)
    xgb_model.fit(X_cv, y_cv)

    write_accuracies(xgb_model, filename, X_train, y_train, X_test, y_test, X_val, y_val)
    with open(filename, 'a') as f:
        with redirect_stdout(f):
            print(xgb_model.best_params_)

def f_lgbm(X_train, y_train, X_test, y_test, X_val, y_val, filename, subpart='f'):
    params = {
        'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450],
        'subsample': [0.4, 0.5, 0.6, 0.7, 0.8],
        'max_depth': [40, 50, 60, 70],
    }

    cv = np.concatenate((-1*np.ones((X_train.shape[0], 1)), np.zeros((X_val.shape[0], 1))), axis=0)
    cv = PredefinedSplit(cv)
    lgbm_model = GridSearchCV(estimator=LGBMClassifier(random_state=0, n_jobs=-1), param_grid=params, cv=cv, scoring='accuracy', n_jobs=-1, refit=True, error_score='raise')
    
    X_cv = vstack([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val], axis=0)
    lgbm_model.fit(X_cv, y_cv)

    write_accuracies(lgbm_model, filename, X_train, y_train, X_test, y_test, X_val, y_val)
    with open(filename, 'a') as f:
        with redirect_stdout(f):
            print(lgbm_model.best_params_)


if __name__=='__main__':
    TRAIN_DATA_PATH = sys.argv[1]
    VAL_DATA_PATH = sys.argv[2]
    TEST_DATA_PATH = sys.argv[3]
    OUTPUT_FOLDER_PATH = sys.argv[4]
    QUESTION_PART = sys.argv[5]

    train_dataset = pd.read_csv(TRAIN_DATA_PATH, na_values=['?'])
    test_dataset = pd.read_csv(TEST_DATA_PATH, na_values=['?'])
    val_dataset = pd.read_csv(VAL_DATA_PATH, na_values=['?'])

    cond_vectorizer = TfidfVectorizer(input='content')
    review_vectorizer = TfidfVectorizer(input='content')
    
    nltk.download('stopwords')
    
    X_train, y_train = get_features(train_dataset, cond_vectorizer, review_vectorizer, method='fit')
    X_test, y_test = get_features(test_dataset, cond_vectorizer, review_vectorizer)
    X_val, y_val = get_features(val_dataset, cond_vectorizer, review_vectorizer)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)


    if QUESTION_PART == 'a':
        with open(os.path.join(OUTPUT_FOLDER_PATH, '2_a.txt'), 'w') as f:
            f.write('')
        a_decision_tree(X_train, y_train, X_test, y_test, X_val, y_val, os.path.join(OUTPUT_FOLDER_PATH, '2_a.txt'))

    if QUESTION_PART == 'b':
        with open(os.path.join(OUTPUT_FOLDER_PATH, '2_b.txt'), 'w') as f:
            f.write('')
        b_dt_gridsearch(X_train, y_train, X_test, y_test, X_val, y_val, os.path.join(OUTPUT_FOLDER_PATH, '2_b.txt'))
        

    if QUESTION_PART == 'c':
        with open(os.path.join(OUTPUT_FOLDER_PATH, '2_c.txt'), 'w') as f:
            f.write('')
        c_ccp(X_train, y_train, X_test, y_test, X_val, y_val, os.path.join(OUTPUT_FOLDER_PATH, '2_c.txt'), OUTPUT_FOLDER_PATH)

    if QUESTION_PART == 'd':
        with open(os.path.join(OUTPUT_FOLDER_PATH, '2_d.txt'), 'w') as f:
            f.write('')
        d_random_forest(X_train, y_train, X_test, y_test, X_val, y_val, os.path.join(OUTPUT_FOLDER_PATH, '2_d.txt'))
    
    if QUESTION_PART == 'e':
        with open(os.path.join(OUTPUT_FOLDER_PATH, '2_e.txt'), 'w') as f:
            f.write('')
        e_xgb(X_train, y_train, X_test, y_test, X_val, y_val, os.path.join(OUTPUT_FOLDER_PATH, '2_e.txt'))

    if QUESTION_PART == 'f':
        with open(os.path.join(OUTPUT_FOLDER_PATH, '2_f.txt'), 'w') as f:
            f.write('')
        f_lgbm(X_train, y_train, X_test, y_test, X_val, y_val, os.path.join(OUTPUT_FOLDER_PATH, '2_f.txt'))

    if QUESTION_PART == 'g':
        lens = [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000]

        m = X_train.shape[0]
        filename = os.path.join(OUTPUT_FOLDER_PATH, '2_g.txt')
        with open(filename, 'w') as f:
            f.write('')
        for l in lens:
            if l > m:
                break

            perm = np.random.permutation(m)
            X_train_sampled, y_train_sampled = X_train[perm[:l]], y_train[perm[:l]]
            a_decision_tree(X_train_sampled, y_train_sampled, X_test, y_test, X_val, y_val, filename, subpart='g'+str(l))
            b_dt_gridsearch(X_train_sampled, y_train_sampled, X_test, y_test, X_val, y_val, filename, subpart='g'+str(l))
            c_ccp(X_train_sampled, y_train_sampled, X_test, y_test, X_val, y_val, filename, OUTPUT_FOLDER_PATH, subpart='g'+str(l))
            d_random_forest(X_train_sampled, y_train_sampled, X_test, y_test, X_val, y_val, filename, subpart='g'+str(l))
            e_xgb(X_train_sampled, y_train_sampled, X_test, y_test, X_val, y_val, filename, subpart='g'+str(l))
            f_lgbm(X_train_sampled, y_train_sampled, X_test, y_test, X_val, y_val, filename, subpart='g'+str(l))