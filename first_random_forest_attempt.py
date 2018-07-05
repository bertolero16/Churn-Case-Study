import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, roc_auc_score

df_train = pd.read_csv('data/churn_train.csv', parse_dates=['last_trip_date', 'signup_date'])
df_test = pd.read_csv('data/churn_test.csv', parse_dates=['last_trip_date', 'signup_date'])
df_tot = pd.read_csv('data/churn.csv', parse_dates=['last_trip_date', 'signup_date'])

cutoff = cutoff = df_tot.last_trip_date.max() - pd.to_timedelta(30,'D')
def prepare_data(df):
    missing = df.isna().sum()
    clean_labels = missing[missing == 0].index
    dirtyl_labels = missing[missing > 0].index
    new_df = df[clean_labels]
    
    to_keep = ['avg_dist', 'avg_surge', 'surge_pct', 'weekday_pct']
    filtered_df = new_df.loc[:, to_keep]
    
    y = df.last_trip_date <= cutoff
    return filtered_df, y

train_X, train_y = prepare_data(df_train)
test_X, test_y = prepare_data(df_test)

rf = RandomForestClassifier()
model = rf.fit(train_X, train_y)
yhat = model.predict(test_X)

def ezprint_cmatrix(ytrue, yhat):
    print("Precision: ", precision_score(ytrue, yhat))
    print("Recall: ", recall_score(ytrue, yhat))
    print("Accuracy score", accuracy_score(ytrue, yhat))

ezprint_cmatrix(test_y, yhat)
ezprint_cmatrix(test_y, np.ones((len(yhat), 1)))


def specificity_score(y_true, y_pred):
    TN = len(y_true[y_true == y_pred and y_true == False])
    FP = len(y_true[y_true != y_pred and y_true == False])
    return TN / (TN + FP)
