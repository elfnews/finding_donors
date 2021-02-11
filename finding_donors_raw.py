import sys
from time import time

import numpy as np
import pandas as pd
import sklearn
from IPython.core.display import display
from sklearn import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, fbeta_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

pd.options.display.max_columns = 200
pd.options.display.max_colwidth = None
pd.options.display.expand_frame_repr = False


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    :param learner: the learning algorithm to be trained and predicted on
    :param sample_size: the size of samples (number) to be drawn from training set
    :param X_train: features training set
    :param y_train: income training set
    :param X_test: features testing set
    :param y_test: income testing set
    :return:
    '''
    print("{} training on {} samples.".format(learner.__class__.__name__, sample_size))
    results = {}
    X_train_sample = X_train[:sample_size]
    y_train_sample = y_train[:sample_size]
    print("{} sample size.".format(sample_size))
    print("X_train")
    display(X_train.head(5))
    print("X_train_sample")
    display(X_train_sample.head(5))
    print("y_train")
    display(y_train.head(5))
    print("y_train_sample")
    display(y_train_sample.head(5))
    print("y_test")
    display(y_train_sample.head(5))

    start = time()
    learner = learner.fit(X_train_sample, y_train_sample)
    end = time()

    results['train_time'] = end - start

    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train_sample)
    end = time()

    results['pred_time'] = end - start
    results['acc_train'] = accuracy_score(y_true=y_train_sample, y_pred=predictions_train)
    results['acc_test'] = accuracy_score(y_true=y_test, y_pred=predictions_test)
    results['f_train'] = fbeta_score(y_true=y_train_sample, y_pred=predictions_train, beta=0.5)
    results['f_test'] = fbeta_score(y_true=y_test, y_pred=predictions_test, beta=0.5)

    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    return results


data = pd.read_csv("census.csv")
display(data.head(n=5))

n_records = len(data)
n_greater_50k = len(data[data['income'] == '>50K'])
n_at_most_50k = len(data[data['income'] == '<=50K'])
greater_percent = n_greater_50k / n_records * 100

print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))

income_raw = data['income']
display(income_raw.head(n=5))
features_raw = data.drop('income', axis=1)
display(features_raw.head(n=5))

skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data=features_raw)
display(features_log_transformed.head(n=5))
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
display(features_log_transformed.head(n=5))

scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
display(features_log_minmax_transform.head(n=5))

features_final = pd.get_dummies(features_log_minmax_transform)
display(features_final.head(n=5))
display(income_raw.head(n=5))
income = income_raw.map({'<=50K': 0, '>50K': 1})
display(income.head(n=5))
encoded = list(features_final.columns)
print(encoded)

X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    income,
                                                    test_size=0.2,
                                                    random_state=0)
print("X_train")
display(X_train.head(n=5))
print("X_test")
display(X_test.head(n=5))
print("y_train")
display(y_train.head(n=5))
print("y_test")
display(y_test.head(n=5))

print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

TP = np.sum(income)
FP = income.count() - TP
print('True Positives:', TP)
print('False Positives:', FP)
TN = 0
FN = 0

accuracy = TP / (TP + FP)
recall = TP / (TP + FN)
precision = TP / (TP + FP)

fscore = (1 + 0.5 ** 2) * (precision * recall) / ((0.5 ** 2 * precision) + recall)
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

clf_A = GaussianNB()
clf_B = DecisionTreeClassifier(random_state=42)
clf_C = LogisticRegression(max_iter=250, random_state=42)

samples_100 = len(X_train)
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.01)

print("Samples sizes are 100%[{}] 10%[{}] 1%[{}]".format(samples_100, samples_10, samples_1))

results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)
        print(results[clf_name][i])
display(results)

print("Python version[{}]".format(sys.version))
print("Sklearn version[{}]".format(sklearn.__version__))

print("Tuning the model ...")
start = time()
# Picked the best of 3 models above
clf = DecisionTreeClassifier(random_state=42)

# Parameters to run grid search on
parameters = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}

#     {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# }


# Scorer to use in grid search
scorer = make_scorer(fbeta_score, beta=0.5)

# Create the grid search object
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# Search and find the best params
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

end = time()
print("Tuning completed in [{}]ms".format(end - start))

print("Original model:")
display(clf)
print("Tuned model:")
display(best_clf)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta=0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5)))

display(best_clf.feature_importances_)
model = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
importances = model.feature_importances_
display(importances)

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

clf = (clone(best_clf)).fit(X_test_reduced, y_train)

reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both version of data
print("Final Model trained on full data\n-----")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5)))
print("Final Model trained on reduced data\n-----")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta=0.5)))
