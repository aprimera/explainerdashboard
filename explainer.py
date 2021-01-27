import pandas as pd
from sklearn.preprocessing import normalize, minmax_scale, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, auc
from tensorflow.keras import optimizers
import numpy as np
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive, titanic_names
import matplotlib.pyplot as plt

feature_descriptions = {
    'CreditScore': "Credit score is a measure of an individual's ability to pay back the borrowed amount.",
    'Geography_Spain': 'Geography_Spain',
    'Geography_Germany': 'Geography_Germany',
    'Gender_Male': 'The gender of the customer.',
    'Age': 'Age of the customer.',
    'Tenure': 'The period of time a customer has been associated with the bank.',
    #Balance: The account balance (the amount of money deposited in the bank account) of the customer.
    'NumOfProducts': 'How many accounts, bank account affiliated products the person has.',
    'HasCrCard': 'Does the customer have a credit card through the bank?',
    'IsActiveMember': 'Subjective, but for the concept',
    'EstimatedSalary': 'Estimated salary of the customer.',
    'Category_Balance': 'Category_Balance'
}

df = pd.read_csv('bank.csv')
test_names = df['RowNumber'].values
columns_todrop = ['RowNumber', 'CustomerId']
surnames = df['Surname']
df = df.drop(columns=columns_todrop)
def cat_bal(x):
    cat = 0
    if x > 0:
        cat = 1
    return cat

df['Category_Balance'] = df['Balance']
df['Category_Balance'] = df['Category_Balance'].apply(cat_bal)
columns_todrop = ['EstimatedSalary']
df = df.drop(columns=columns_todrop)
df_onehotenc = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
features = list(df_onehotenc.columns)
features.remove('Exited')
predictor = 'Exited'
X = df_onehotenc[features]
y = df_onehotenc[predictor]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#print(type(X_train))
#exit(0)
test_names = X_test['Surname'].values
train_names = X_train['Surname'].values
X_train = X_train.drop(columns=['Surname'])
X_test = X_test.drop(columns=['Surname'])
sc=StandardScaler()

cols = list(X_train.columns)
input_size = len(cols)
X_train = sc.fit_transform(X_train)
X_train = pd.DataFrame(columns=cols, data=X_train)
X_train.index = train_names
X_test = sc.fit_transform(X_test)
X_test = pd.DataFrame(columns=cols, data=X_test)
X_test.index = test_names

model = Sequential()
model.add(Dense(8, input_shape = (input_size,), activation = 'tanh'))
model.add(Dense(8, activation = 'tanh'))
model.add(Dense(1, activation = 'sigmoid'))
sgd = optimizers.Adam(lr = 0.001)
model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train.values, batch_size = 200, epochs = 200, verbose = 1)


#X_train, y_train, X_test, y_test = titanic_survive()
#train_names, test_names = titanic_names()
#model = RandomForestClassifier(n_estimators=50, max_depth=5)
#model.fit(X_train, y_train)

explainer = ClassifierExplainer(model, X_test, y_test, 
                                #cats=[{'Gender': ['Gender_Male']},
                                #      {'Geography': ['Geography_Spain', 'Geography_Germany']}
                                #     ],
                                descriptions=feature_descriptions, # defaults to None
                                labels=['Not Exited', 'Exited'], # defaults to ['0', '1', etc]
                                idxs = test_names, # defaults to X.index
                                index_name = 'Surname', # defaults to X.index.name
                                target = "Exited", # defaults to y.name
                                X_background = X_test,
                                shap='deep',
                                )

db = ExplainerDashboard(explainer, 
                        title="Bank Churn", # defaults to "Model Explainer"
                        whatif=False, # you can switch off tabs with bools
                        shap_dependence=False,
                        shap_interaction=False,
                        decision_trees=False
                        )
db.run(port=8050)