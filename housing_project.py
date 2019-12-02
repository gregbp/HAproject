import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score, f1_score, recall_score

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)



# Evaluating our model by splitting the train set into train and test sets (70-30)
def ModelEvaluation(train):

    # New train and test sets
    X = train.drop('IS_SCAMMER', axis=1)
    y = train['IS_SCAMMER']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    # Dictionary that stores the algorithm name (key) and its precision (value)
    f1_scores = {}

    # Dataframe that stores the metric results for every algorithm
    data = {'F1_score': [0.0, 0.0, 0.0, 0.0, 0.0],
            'Recall Score': [0.0, 0.0, 0.0, 0.0, 0.0],
            'Average Precision Score': [0.0, 0.0, 0.0, 0.0, 0.0],
            'Accuracy Score': [0.0, 0.0, 0.0, 0.0, 0.0],
            }
    metrics = pd.DataFrame(data)
    metrics.index = ['Decision Tree', 'Random Forest', 'KNN', 'SVM', 'Logistic Regression']


    # DecisionTree
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    predictions = dtree.predict(X_test)

    f1_scores.update({'Decision Tree': f1_score(y_test, predictions)})

    metrics.iloc[0] = [f1_score(y_test, predictions), recall_score(y_test, predictions),
                       average_precision_score(y_test, predictions), accuracy_score(y_test, predictions)]


    # RandomForest
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)

    f1_scores.update({'Random Forest': f1_score(y_test, predictions)})

    metrics.iloc[1] = [f1_score(y_test, predictions), recall_score(y_test, predictions),
                       average_precision_score(y_test, predictions), accuracy_score(y_test, predictions)]


    # KNN
    # Finding the best K
    max_f1_scr = -1.0
    max_k = -1
    for k in range(7, 10):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)

        f1_scr = f1_score(y_test, predictions)

        if f1_scr > max_f1_scr:
            max_f1_scr = f1_scr
            max_k = k

    # Fitting a model using the found best K
    knn = KNeighborsClassifier(n_neighbors=max_k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    f1_scores.update({'KNN': f1_score(y_test, predictions)})

    metrics.iloc[2] = [f1_score(y_test, predictions), recall_score(y_test, predictions),
                       average_precision_score(y_test, predictions), accuracy_score(y_test, predictions)]



    # SVM
    model = SVC()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    f1_scores.update({'SVM': f1_score(y_test, predictions)})
     
    metrics.iloc[3] = [f1_score(y_test, predictions), recall_score(y_test, predictions),         
                       average_precision_score(y_test, predictions), accuracy_score(y_test, predictions)]



    # LogisticRegression
    logmodel = LogisticRegression(solver='lbfgs')
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)

    f1_scores.update({'Logistic Regression': f1_score(y_test, predictions)})

    metrics.iloc[4] = [f1_score(y_test, predictions), recall_score(y_test, predictions),
                       average_precision_score(y_test, predictions), accuracy_score(y_test, predictions)]




    # Printing the metric results for every algorithm
    print("\n\n\tmetrics DF inside function\n",metrics,"\n\n")
    print('f1_scores -> ', f1_scores)

    # Finding the name of the best algorithm (using finally f1 score as a measure
    best_alg = max(f1_scores, key=f1_scores.get)
    print('best_alg ', best_alg)

    # Returning the name of best algorithm (best_alg), its f1 score
    # (f1_scores[best_alg]) and best K for KNN algorithm (max_k)
    return best_alg, f1_scores[best_alg], max_k






# ---------------- MAIN PROGRAM ----------------

# Directory for the input & output files
dir_path = os.path.dirname(os.path.realpath(__file__))
inout_path = dir_path+'\\inoutput'

# Initial train and test dataframes
train = pd.read_csv(inout_path + "\\housing_data_train.csv")
test = pd.read_csv(inout_path + "\\housing_data_test.csv")

# Keeping in the train set the same columns that exist in the test set + 'IS_SCAMMER' column
test_cols = test.columns.tolist()
test_cols.append('IS_SCAMMER')
train = train[test_cols]


# SAMPLE PROCESSING -> dropping only the columns 'BROWSER', 'OS', 'ANONYMISED_EMAIL' and
# feature engineering on the columns 'LOGIN_COUNTRY_CODE', 'LISTING_COUNTRY_CODE' and 'LISTING_CITY'

# Convert boolean values to integer (True -> 1, False -> 0)
train['MANAGED_ACCOUNT'] = train['MANAGED_ACCOUNT'].apply(lambda row: 1 if row == 'True' else 0)
test['MANAGED_ACCOUNT'] = test['MANAGED_ACCOUNT'].apply(lambda row: 1 if row == 'True' else 0)


# Dropping the columns 'BROWSER', 'OS', 'ANONYMISED_EMAIL'
dropped_cols = ['BROWSER', 'OS', 'ANONYMISED_EMAIL']
train = train.drop(dropped_cols, axis=1)
test = test.drop(dropped_cols, axis=1)


# Filling nan values of column 'LOGIN_COUNTRY_CODE' with the values of 'LISTING_COUNTRY_CODE' at the same row
nan_indexes = list(train['LOGIN_COUNTRY_CODE'].index[train['LOGIN_COUNTRY_CODE'].isnull()])
for i in nan_indexes:
    train['LOGIN_COUNTRY_CODE'][i] = train['LISTING_COUNTRY_CODE'][i]

nan_indexes = list(test['LOGIN_COUNTRY_CODE'].index[test['LOGIN_COUNTRY_CODE'].isnull()])
for i in nan_indexes:
    test['LOGIN_COUNTRY_CODE'][i] = test['LISTING_COUNTRY_CODE'][i]


# Mapping 'LOGIN_COUNTRY_CODE' & 'LISTING_COUNTRY_CODE' values to integer numbers
countries = train['LOGIN_COUNTRY_CODE'].unique().tolist() + test['LOGIN_COUNTRY_CODE'].unique().tolist() + train[
        'LISTING_COUNTRY_CODE'].unique().tolist() + test['LISTING_COUNTRY_CODE'].unique().tolist()
countries_list = list(set(countries))
countries_dict = dict()
for value in range(0, len(countries_list)):
    countries_dict.update({countries_list[value]: value})

# Mapping for train set
i = 0
for x in train['LOGIN_COUNTRY_CODE']:
    train['LOGIN_COUNTRY_CODE'][i] = countries_dict[x]
    train['LISTING_COUNTRY_CODE'][i] = countries_dict[x]
    i = i + 1

# Mapping for test set
i = 0
for x in test['LOGIN_COUNTRY_CODE']:
    test['LOGIN_COUNTRY_CODE'][i] = countries_dict[x]
    test['LISTING_COUNTRY_CODE'][i] = countries_dict[x]
    i = i + 1


# Mapping LISTING_CITY values to integer numbers
cities = train['LISTING_CITY'].unique().tolist() + test['LISTING_CITY'].unique().tolist()
cities_list = list(set(cities))

cities_dict = dict()
for value in range(0, len(cities_list)):
    cities_dict.update({cities_list[value]: value})

# Mapping for train set
i = 0
for x in train['LISTING_CITY']:
    train['LISTING_CITY'][i] = cities_dict[x]
    i = i + 1

# Mapping for test set
i = 0
for x in test['LISTING_CITY']:
    test['LISTING_CITY'][i] = cities_dict[x]
    i = i + 1




# MODEL EVALUATION
# Model Evaluation results -> name of best algorithm (alg), its f1 score (alg_f1scr) and best K for KNN algorithm (maxK)
alg, alg_f1scr, maxK = ModelEvaluation(train)
print("\n\talg", alg, alg_f1scr, maxK, '\n\n')




# MAKING PREDICTIONS
X = train.drop('IS_SCAMMER', axis=1)
y = train['IS_SCAMMER']

# Choosing the best algorithm after model evaluation (line 217)
if alg == 'Decision Tree':
    dtree = DecisionTreeClassifier()
    dtree.fit(X, y)
    predictions = dtree.predict(test)
    test['IS_SCAMMER'] = predictions

if alg == 'Random Forest':
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X, y)
    predictions = rfc.predict(test)
    test['IS_SCAMMER'] = predictions

if alg == 'KNN':
    knn = KNeighborsClassifier(n_neighbors=max_k)
    knn.fit(X, y)
    predictions = knn.predict(test)
    test['IS_SCAMMER'] = predictions

if alg == 'SVM':
    model = SVC()
    model.fit(X, y)
    predictions = model.predict(test)
    test['IS_SCAMMER'] = predictions

if alg == 'Logistic Regression':
    logmodel = LogisticRegression(solver='lbfgs')
    logmodel.fit(X, y)
    predictions = logmodel.predict(test)
    test['IS_SCAMMER'] = predictions

# Exporting the predictions to csv file
export_csv = test.to_csv(inout_path + "\\housing_data_test_labeled.csv", index=None, header=True)


# Testing the predictions on the real data
test_full = pd.read_csv(inout_path + "\\housing_data_test_full.csv")

y_test_full = test_full['IS_SCAMMER']
data = {'F1_score':[0.0],
        'Recall Score':[0.0],
        'Average Precision Score':[0.0],
        'Accuracy Score':[0.0],
        }

metrics = pd.DataFrame(data)
metrics.index = [alg]
metrics.iloc[0] = [f1_score(y_test_full, predictions), recall_score(y_test_full, predictions),
                       average_precision_score(y_test_full, predictions), accuracy_score(y_test_full, predictions)]

# Printing the results of the chosen algorithm
print('\n\n\n\tMetrics\n\n', metrics)
print('\n\n\tConfusion Matrix\n\n', confusion_matrix(y_test_full, predictions))
print('\n\n\tClassification Report\n\n', classification_report(y_test_full, predictions))
