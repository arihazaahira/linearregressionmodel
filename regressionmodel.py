from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import math
df = pd.read_csv('newdata.csv')

X=df[['Open','High','Low','Volume']].values #indep variables
Y=df['Close'].values #dep var 
#tesing 80%train and 20% testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=0)

regressor = LinearRegression()
model = regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
predicted=regressor.predict(X_test)
print(predicted.shape)# va afficher le meme nbr de donnes qu ona utiliser pour le test nbrdepred=nbrdetest
#en cherche les coefs a et b tel que y=ax+b
print("model coeff:",regressor.coef_)#represente limportance de chaque var dans la var predite (dependante)
print("model intercept:",regressor.intercept_)#represente lq valeur de y lorsque tous les autres vqr sont nulles

dframe=pd.DataFrame(Y_test,predicted)
dfr=pd.DataFrame({'Actuel_price':Y_test,'Predicted_price':predicted})
print(dfr)
#pour avoir la precision du model
from sklearn.metrics import confusion_matrix,accuracy_score
regression_confidence=regressor.score(X_test,Y_test)
print("linear regression confidence:",regression_confidence)
x2 = abs(predicted - Y_test)#err absolue
y2 = 100 * (x2 / Y_test)#err relative
accuracy = round(100 - np.mean(y2), 2)  
print('accuracy:', accuracy, '%')
plt.scatter(dfr.Actuel_price,dfr.Predicted_price,color='darkblue')
plt.xlabel("Actual Price")
plt.ylabel("Predicted price")
plt.show()
graph=dfr.head(15)
graph.plot(kind='bar')
plt.show()
