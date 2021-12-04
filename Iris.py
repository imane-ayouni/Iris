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

from sklearn import model_selection

models = []
models.append(("LogReg", LogisticRegression()))
models.append(("RanFor", RandomForestClassifier()))
models.append(("DecTree", DecisionTreeClassifier()))

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



