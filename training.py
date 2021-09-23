import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == "__main__":
    # Reading the data
    df = pd.read_csv("data.csv")
    df.drop(['test_date'], axis=1, inplace=True)
    df = df.dropna()
    df = df.reindex(columns = ['cough','fever','sore_throat', 'shortness_of_breath', 'head_ache', 'age_60_and_above', 'gender', 'test_indication', 'corona_result'])
    df_filtered = df[df['corona_result'] != "other"]
    df = df_filtered
    # Replacing positivie with 1 and negative with 0
    df.loc[(df.corona_result == "negative"), 'corona_result'] = int(0)
    df.loc[(df.corona_result == "positive"), 'corona_result'] = int(1)
    df.loc[(df.age_60_and_above == "No"), 'age_60_and_above'] = int(0)
    df.loc[(df.age_60_and_above == "Yes"), 'age_60_and_above'] = int(1)
    df.loc[(df.test_indication == "Other"), 'test_indication'] = int(1)
    df.loc[(df.test_indication == "Contact with confirmed"), 'test_indication'] = int(2)
    df.loc[(df.test_indication == "Abroad"), 'test_indication'] = int(3)
    df.loc[(df.gender == "male"), 'gender'] = int(0)
    df.loc[(df.gender == "female"), 'gender'] = int(1)
    df["corona_result"] = df["corona_result"].astype(str).astype(int)
    df["age_60_and_above"] = df["age_60_and_above"].astype(str).astype(int)
    df["test_indication"] = df["test_indication"].astype(str).astype(int)
    df["gender"] = df["gender"].astype(str).astype(int)

    train, test = data_split(df, 0.2)
    X_train = train[['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 'age_60_and_above', 'gender', 'test_indication']].to_numpy()
    X_test = test[['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 'age_60_and_above', 'gender', 'test_indication']].to_numpy()
    Y_train = train['corona_result'].to_numpy().reshape(1721519,)
    Y_test = test['corona_result'].to_numpy().reshape(430379,)

    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    # open a file, where we want to store the data
    file = open('model.pkl', 'wb')

    # dump information to the file
    pickle.dump(clf, file)
    file.close()
    
    # Predicting the test set results and calculating the accuracy
    y_pred = clf.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, Y_test)))

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(Y_test, y_pred)
    print(confusion_matrix)
    # The result is telling us that we have 630036 + 18624 correct predictions and 50055 + 11411 incorrect predictions.

    
    

