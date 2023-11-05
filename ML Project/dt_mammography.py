import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


def preprocess_dropna(dataset):
    dataset.dropna(axis=0, inplace=True)
    dataset = dataset.astype('int64')
    X, y = dataset.iloc[:, 1:-1], dataset.iloc[:, -1]

    return X, y


def preprocess_impute(imputer, dataset):
    dataset = imputer.transform(dataset)
    X, y = dataset[:, 1:-1], dataset[:, -1]

    return X, y

def preprocess(dataset):
    X, y = dataset.iloc[:, 1:-1], dataset.iloc[:, -1]

    return X, y

def write_accuracies(model, filename, X_train, y_train, X_test, y_test, X_val, y_val):
    with open(filename, 'a+') as f:
        with redirect_stdout(f):
            print('Training Accuracy: ', model.score(X_train, y_train))
            print('Testing Accuracy: ', model.score(X_test, y_test))
            print('Validation Accuracy: ', model.score(X_val, y_val))


def a_decision_tree(X_train, y_train, X_test, y_test, X_val, y_val, filename, OUTPUT_FOLDER_PATH, subpart='a'):
    dt_model = DecisionTreeClassifier(random_state=0)
    dt_model.fit(X_train, y_train)
    write_accuracies(dt_model, filename, X_train, y_train, X_test, y_test, X_val, y_val)

    plot_tree(dt_model)
    plt.savefig(os.path.join(OUTPUT_FOLDER_PATH, '1_'+subpart+'_dtree.png'), dpi=300)

def b_dt_gridsearch(X_train, y_train, X_test, y_test, X_val, y_val, filename, OUTPUT_FOLDER_PATH, subpart='b'):
    params = {
        'max_depth': [4, 5, 6, 8, 10, 12, 15, 18, 20],
        'min_samples_split': [2, 3, 4, 5, 6, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    cv = np.concatenate((-1*np.ones((X_train.shape[0], 1)), np.zeros((X_val.shape[0], 1))), axis=0)
    cv = PredefinedSplit(cv)
    gs_model = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=params, cv=cv, scoring='accuracy', n_jobs=-1, refit=True, error_score='raise')

    X_cv = np.concatenate([X_train, X_val], axis=0)
    y_cv = np.concatenate([y_train, y_val], axis=0)

    gs_model.fit(X_cv, y_cv)

    write_accuracies(gs_model, filename, X_train, y_train, X_test, y_test, X_val, y_val)
    with open(filename, 'a') as f:
        f.write(str(gs_model.best_params_))

    plot_tree(gs_model.best_estimator_)
    plt.savefig(os.path.join(OUTPUT_FOLDER_PATH, '1_'+subpart+'_dtree_gs.png'), dpi=300)

def c_ccp(X_train, y_train, X_test, y_test, X_val, y_val, filename, OUTPUT_FOLDER_PATH, subpart='c'):
    ccp_path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = ccp_path.ccp_alphas, ccp_path.impurities

    # print(ccp_alphas.shape)
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle='steps-post')
    ax.set_xlabel('effective alpha')
    ax.set_ylabel('total impurity of leaves')
    ax.set_title('Total Impurity vs effective alpha for training set')
    fig.savefig(os.path.join(OUTPUT_FOLDER_PATH, '1_'+subpart+'_alp_vs_imp.png'), dpi=300)

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
    fig.savefig(os.path.join(OUTPUT_FOLDER_PATH, '1_'+subpart+'_nodes_vs_alpha_depth_vs_alpha.png'), dpi=300)

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
    plt.savefig(os.path.join(OUTPUT_FOLDER_PATH, '1_'+subpart+'_accuracy_vs_alpha.png'), dpi=300)

    ccp_model = ccp_models[np.argmax(val_scores)]
    plot_tree(ccp_model)
    plt.savefig(os.path.join(OUTPUT_FOLDER_PATH, '1_'+subpart+'_dtree_ccp.png'), dpi=300)
    write_accuracies(ccp_model, filename, X_train, y_train, X_test, y_test, X_val, y_val)

def d_random_forest(X_train, y_train, X_test, y_test, X_val, y_val, filename, subpart='d'):
    params = {
        'n_estimators': [10, 50, 100, 120, 150, 180, 200],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 4, 6, 8, 10, 12],
    }

    cv = np.concatenate((-1*np.ones((X_train.shape[0], 1)), np.zeros((X_val.shape[0], 1))), axis=0)
    cv = PredefinedSplit(cv)
    rf_model = GridSearchCV(estimator=RandomForestClassifier(oob_score=True, random_state=0), param_grid=params, cv=cv, scoring='accuracy', n_jobs=-1, refit=True, error_score='raise')
    
    X_cv = np.concatenate([X_train, X_val], axis=0)
    y_cv = np.concatenate([y_train, y_val], axis=0)
    rf_model.fit(X_cv, y_cv)

    write_accuracies(rf_model, filename, X_train, y_train, X_test, y_test, X_val, y_val)
    with open(filename, 'a') as f:
        with redirect_stdout(f):
            print('OOB Accuracy: ', rf_model.best_estimator_.oob_score_)
            print(rf_model.best_params_)


if __name__=='__main__':
    TRAIN_DATA_PATH = sys.argv[1]
    VAL_DATA_PATH = sys.argv[2]
    TEST_DATA_PATH = sys.argv[3]
    OUTPUT_FOLDER_PATH = sys.argv[4]
    QUESTION_PART = sys.argv[5]

    train_dataset = pd.read_csv(TRAIN_DATA_PATH, na_values=['?'])
    test_dataset = pd.read_csv(TEST_DATA_PATH, na_values=['?'])
    val_dataset = pd.read_csv(VAL_DATA_PATH, na_values=['?'])

    X_train, y_train = preprocess_dropna(train_dataset.copy())
    X_test, y_test = preprocess_dropna(test_dataset.copy())
    X_val, y_val = preprocess_dropna(val_dataset.copy())

    if QUESTION_PART == 'a':
        with open(os.path.join(OUTPUT_FOLDER_PATH, '1_a.txt'), 'w') as f:
            f.write('')
        a_decision_tree(X_train, y_train, X_test, y_test, X_val, y_val, os.path.join(OUTPUT_FOLDER_PATH, '1_a.txt'), OUTPUT_FOLDER_PATH)

    if QUESTION_PART == 'b':
        with open(os.path.join(OUTPUT_FOLDER_PATH, '1_b.txt'), 'w') as f:
            f.write('')
        b_dt_gridsearch(X_train, y_train, X_test, y_test, X_val, y_val, os.path.join(OUTPUT_FOLDER_PATH, '1_b.txt'), OUTPUT_FOLDER_PATH)
        

    if QUESTION_PART == 'c':
        with open(os.path.join(OUTPUT_FOLDER_PATH, '1_c.txt'), 'w') as f:
            f.write('')
        c_ccp(X_train, y_train, X_test, y_test, X_val, y_val, os.path.join(OUTPUT_FOLDER_PATH, '1_c.txt'), OUTPUT_FOLDER_PATH)

    if QUESTION_PART == 'd':
        with open(os.path.join(OUTPUT_FOLDER_PATH, '1_d.txt'), 'w') as f:
            f.write('')
        d_random_forest(X_train, y_train, X_test, y_test, X_val, y_val, os.path.join(OUTPUT_FOLDER_PATH, '1_d.txt'))

    if QUESTION_PART == 'e':
        imp_median = SimpleImputer(strategy='median', missing_values=pd.NA)
        imp_mode = SimpleImputer(strategy='most_frequent', missing_values=pd.NA)

        imp_median.fit(train_dataset)
        imp_mode.fit(train_dataset)

        X_train, y_train = preprocess_impute(imp_median, train_dataset.copy())
        X_test, y_test = preprocess_impute(imp_median, test_dataset.copy())
        X_val, y_val = preprocess_impute(imp_median, val_dataset.copy())

        filename = os.path.join(OUTPUT_FOLDER_PATH, '1_e.txt')
        with open(filename, 'w') as f:
            f.write('')
        a_decision_tree(X_train, y_train, X_test, y_test, X_val, y_val, filename, OUTPUT_FOLDER_PATH, subpart='e_median')
        b_dt_gridsearch(X_train, y_train, X_test, y_test, X_val, y_val, filename, OUTPUT_FOLDER_PATH, subpart='e_median')
        c_ccp(X_train, y_train, X_test, y_test, X_val, y_val, filename, OUTPUT_FOLDER_PATH, subpart='e_median')
        d_random_forest(X_train, y_train, X_test, y_test, X_val, y_val, filename, subpart='e_median')

        X_train, y_train = preprocess_impute(imp_mode, train_dataset.copy())
        X_test, y_test = preprocess_impute(imp_mode, test_dataset.copy())
        X_val, y_val = preprocess_impute(imp_mode, val_dataset.copy())

        a_decision_tree(X_train, y_train, X_test, y_test, X_val, y_val, filename, OUTPUT_FOLDER_PATH, subpart='e_mode')
        b_dt_gridsearch(X_train, y_train, X_test, y_test, X_val, y_val, filename, OUTPUT_FOLDER_PATH, subpart='e_mode')
        c_ccp(X_train, y_train, X_test, y_test, X_val, y_val, filename, OUTPUT_FOLDER_PATH, subpart='e_mode')
        d_random_forest(X_train, y_train, X_test, y_test, X_val, y_val, filename, subpart='e_mode')

    if QUESTION_PART == 'f':
        X_train, y_train = preprocess(train_dataset)
        X_test, y_test = preprocess(test_dataset)
        X_val, y_val = preprocess(val_dataset)
        
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        y_val = le.transform(y_val)
        

        params = {
            'n_estimators': [10, 20, 30, 40, 50],
            'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'max_depth': [4, 5, 6, 7, 8, 9, 10],
        }

        cv = np.concatenate((-1*np.ones((X_train.shape[0], 1)), np.zeros((X_val.shape[0], 1))), axis=0)
        cv = PredefinedSplit(cv)
        xgb_model = GridSearchCV(estimator=XGBClassifier(random_state=0, n_jobs=-1), param_grid=params, cv=cv, scoring='accuracy', n_jobs=-1, refit=True, error_score='raise')
        
        X_cv = np.concatenate([X_train, X_val], axis=0)
        y_cv = np.concatenate([y_train, y_val], axis=0)
        xgb_model.fit(X_cv, y_cv)

        filename = os.path.join(OUTPUT_FOLDER_PATH, '1_f.txt')
        with open(filename, 'w') as f:
            f.write('')
        write_accuracies(xgb_model, filename, X_train, y_train, X_test, y_test, X_val, y_val)
        with open(filename, 'a') as f:
            with redirect_stdout(f):
                print(xgb_model.best_params_)