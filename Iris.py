import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

flowers_data = pd.read_csv(r'C:\Users\imane\OneDrive\Desktop\Iris\Iris.csv', sep =",", encoding = "utf-8")

flowers_data.info()
flowers_data = flowers_data.drop("Id", axis = 1)
print(flowers_data)

flowers_data.hist(bins=70, figsize=(20, 15))
plt.show()

sns.heatmap(flowers_data.corr(),annot = True)
plt.show()

sns.lmplot( x="SepalLengthCm", y="SepalWidthCm", data= flowers_data, fit_reg=False, hue='Species', legend=True)
plt.show()

sns.lmplot( x="PetalLengthCm", y="PetalWidthCm", data= flowers_data, fit_reg=False, hue='Species', legend=True)
plt.show()


sns.violinplot(y = flowers_data["SepalLengthCm"], x = flowers_data["Species"])
plt.show()

sns.violinplot(y = flowers_data["SepalWidthCm"], x = flowers_data["Species"])
plt.show()

sns.violinplot(y = flowers_data["PetalLengthCm"], x = flowers_data["Species"])
plt.show()

sns.violinplot(y = flowers_data["PetalWidthCm"], x = flowers_data["Species"])
plt.show()

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(flowers_data, test_size = 0.2,random_state = 42)

train_set_labels = train_set["Species"].copy()
train_set = train_set.drop("Species", axis = 1)
test_set_labels = test_set["Species"].copy()
test_set = test_set.drop("Species", axis = 1)

from sklearn.preprocessing import MinMaxScaler as Scaler
scaler = Scaler()
scaler.fit(train_set)
scaled_train_set = scaler.transform(train_set)
scaled_test_set = scaler.transform(test_set)

scaled_df = pd.DataFrame(data = scaled_train_set)
print(scaled_df)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from  sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.multiclass import OneVsRestClassifier

models = []
models.append(("LogReg", LogisticRegression()))
models.append(("RanFor", RandomForestClassifier()))
models.append(("DecTree", DecisionTreeClassifier()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("SVC",SVC()))
models.append(("XGB",OneVsRestClassifier(xgb.XGBClassifier(use_label_encoder=False,eval_metric='mlogloss'))))

seed = 7
results = []
names = []
X = scaled_train_set
Y = train_set_labels


for name, model in models:
    kfolds = model_selection.KFold(n_splits=10)
    cv_results  = model_selection.cross_val_score(model, X, Y, cv=kfolds, scoring="accuracy")
    results.append(cv_results)
    names.append(name)

    outcome = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(outcome)



figure = plt.figure()
figure.suptitle("Algorithms performance")
ax = figure.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


petals_data = flowers_data[["PetalLengthCm","PetalWidthCm","Species"]]
train_set_petals, test_set_petals = train_test_split(petals_data, test_size = 0.2,random_state = 42)

train_set_labels_petals = train_set_petals["Species"].copy()
train_set_petals = train_set_petals.drop("Species", axis = 1)
test_set_labels_petals = test_set_petals["Species"].copy()
test_set_petals = test_set_petals.drop("Species", axis = 1)

scaler_petals = Scaler()
scaler_petals.fit(train_set_petals)
scaled_train_set_petals = scaler_petals.transform(train_set_petals)
scaled_test_set_petals = scaler_petals.transform(test_set_petals)

scaled_df_petals = pd.DataFrame(data = scaled_train_set_petals)

models_petals = []
models_petals.append(("LogReg", LogisticRegression()))
models_petals.append(("RanFor", RandomForestClassifier()))
models_petals.append(("DecTree", DecisionTreeClassifier()))
models_petals.append(("KNN",KNeighborsClassifier()))
models_petals.append(("SVC",SVC()))
models_petals.append(("XGB",OneVsRestClassifier(xgb.XGBClassifier(use_label_encoder=False,eval_metric='mlogloss'))))


seed_petals = 7
results_petals = []
names_petals = []
X_petals = scaled_train_set_petals
Y_petals = train_set_labels_petals

for name, model in models_petals:
    kfolds = model_selection.KFold(n_splits=10)
    cv_results_petals  = model_selection.cross_val_score(model, X_petals, Y_petals, cv=kfolds, scoring="accuracy")
    results_petals.append(cv_results_petals)
    names_petals.append(name)

    outcome_petals = "%s: %f (%f)" % (name, cv_results_petals.mean(), cv_results_petals.std())
    print(outcome_petals)

figure_petals = plt.figure()
figure.suptitle("Algorithms performance on Petals data")
ax_petals = figure_petals.add_subplot(111)
plt.boxplot(results_petals)
ax_petals.set_xticklabels(names_petals)
plt.show()