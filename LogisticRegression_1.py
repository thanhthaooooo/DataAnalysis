#%% - Import Lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Function import sigmoid
import statsmodels.api as sm

#%% - Some configs
plt.rcParams['figure.figsize']=(10,8)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 14

#%% -
df= pd.read_csv("./Data/Insurance.csv")


#%%
plt.scatter(df.age,df['bought_insurance'], color="darkgreen", marker="o")
plt.show()

#%% - Slipt train/test dataset
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(df[["age"]],df["bought_insurance"], test_size=0.2)


#%% - Create model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train, y_train)
# logit(odds) = a+b * age

#%%- Get results
intercept = model.intercept_
coef = model.coef_
score = model.score(X_train,y_train)
pro_matrix = model.predict_proba(X_train)

#%% -
from sklearn.metrics import classification_report, confusion_matrix
y_pred =model.predict(X_train)
print(classification_report(y_train,y_pred))
cm = confusion_matrix(y_train,y_pred)

#%% -
plt.scatter(X_train,y_train, color="cyan", marker="o",label='Actual')
plt.scatter(X_train,y_pred, color="cyan", marker="+",label='Predict')
plt.legend()
plt.show()

#%%- Truwj quan ma traanj phaan loaij
fig, ax = plt.subplots()
ax.imshow(cm)
ax.xaxis.set(ticks = (0,1),ticklabels = ('Predicted 0s',"Predicted 1s"))
ax.yaxis.set(ticks = (0,1),ticklabels = ('Actual 0s',"Actual 1s"))
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha = 'center', va = 'center',color ="#FF0000", fontsize = 26)

plt.show()

#%% - Predict
pred_values = model.predict(X_test)
pred_score = model.score(X_test, y_test)
pre_prob_matrix =model.predict_proba(X_test)
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#%% - define prediction function via sigmoid
def prediction_func(age, inter, coef):
    x = inter + coef*age
    return sigmoid(x)
#%% - define prediction function via sigmoid
def prediction_func(age, inter, coef):
    x = inter + coef*age
    return sigmoid(x)

#%% - Draw Sigmoid Plot
plt.scatter(X_train,y_train, color="red", marker='o')
x_test = np.linspace(10,75,25)
sigs = []
inte = intercept[0]
co = coef[0][0]
for item in x_test:
    # print(prediction_func(item, inte, co))
    sigs.append(prediction_func(item,intercept[0],coef[0][0]))
plt.plot(x_test, sigs, color="g")
plt.scatter(X_test,y_test, color="b", s=150, label = 'Actual')
plt.scatter(X_test,pred_values, color="y", s=150, label = 'Predict')
plt.legend(loc="center right")
plt.show()

#%% - predict future values
#pred_prob = prediction_func(56, intercept[0], coefs[0][0])