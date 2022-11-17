import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('Admission_Predict.csv')
print(df.columns)
print(df.shape)
print(df.head())

from sklearn.preprocessing import Binarizer
bi = Binarizer(threshold=0.75)
df['Chance of Admit '] = bi.fit_transform(df[['Chance of Admit ']])
print(df.head())

x = df.drop('Chance of Admit ', axis=1)
y = df['Chance of Admit '] #dropping Chance of Admit from X and making it output Y

y = y.astype('int') #changing the datatype of Y

sns.countplot(x=y) #plotting graph Chance of Admit
plt.show()

print(y.value_counts())

#Cross Validation
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.25)

x_train.shape
x_test.shape


#Applying Machine Learning Algorithm (Decision Tree Classifier)
from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
result = pd.DataFrame({
    'actual' : y_test,
    'predicted' : y_pred
}) 
print(result)

#Evaluation of Model
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import classification_report

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))