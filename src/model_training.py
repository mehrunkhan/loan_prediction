import pickle
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Call our function from data_cleaning file
from data_cleaning import clean_data

# Save clean_data in data
data = clean_data()


# changing categorical column values to numerical binary
code_numeric = {
    "M": 1, "F": 2,  # code gender

    "Secondary or secondary special": 1, "Higher education": 2,  # education type
    "Incomplete higher": 3, "Lower secondary": 4,
    "Academic degree": 5,

    "Married": 1, "Single or not married": 2, "Civil marriage": 3,  # Name family status
    "Separated": 4, "Widow": 5, "Unknown": 6,

    "House or apartment": 1, "With parents": 2,  # name housing type
    "Municipal apartment": 3, "Rented apartment": 4, "Office apartment": 5, "Co-op apartment": 6,

    "Production_Worker": 1, "Laborers": 2, "restaurant worker": 3,  # occupation type
    "Sales staff": 4, "Drivers": 5,
    "High skill tech staff": 6, "Accountants": 7,
    "Medicine staff": 8, "Security staff": 9,
    "Cooking staff": 10, "Cleaning staff": 11,
    "Private service staff": 12, "Low-skill Laborers": 13,
    "Waiters/barmen staff": 14, "Secretaries": 15,
    "HR staff": 16, "IT staff": 17,

    "Y": 1, "N": 0, }
data = data.applymap(lambda s: code_numeric.get(s) if s in code_numeric else s)

data.to_csv("synchronized_data.csv", index=False)

# splitting train and target values in separate dataframe. Our target column is loan_status
x, y = data.drop("loan_status", axis=1), data.loan_status


# x = pd.get_dummies(x)
# splitting test and train dataset to portions for testing and training.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, train_size=.75, random_state=4)

# initiating Logistic Regression here.
logistic_regression = LogisticRegression()
# training initiation
logistic_regression_filename = "../data/logistic_model.pkl"
logistic_regression.fit(x_train, y_train)
with open(logistic_regression_filename, 'wb') as file:
    pickle.dump(logistic_regression, file)

y_pred = logistic_regression.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)

print("Logistic_Regrassion", 100 * accuracy)

# Random_forest
forest = RandomForestClassifier()
# training initiation
forest.fit(x_train, y_train)
with open("../data/random_forest_model.pkl", 'wb') as file:
    pickle.dump(forest, file)

ypred_forest = forest.predict(x_test)
accuracy = metrics.accuracy_score(y_test, ypred_forest)
accuracy_percentage = 100 * accuracy
print("Random_forest", accuracy_percentage)

# Decision_Tree
tree = DecisionTreeClassifier()
# training initiation
tree.fit(x_train, y_train)
with open("../data/decision_tree_model.sav", 'wb') as file:
    pickle.dump(tree, file)
ypred_tree = tree.predict(x_test)
accuracy = metrics.accuracy_score(y_test, ypred_tree)
accuracy_percentage = 100 * accuracy
print("Decision_Tree", accuracy_percentage)
