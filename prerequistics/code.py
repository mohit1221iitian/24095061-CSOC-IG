import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time
import math
import random
plt.style.use('fivethirtyeight')


# -> DATA PREPROCESSING
data=pd.read_csv('housing.csv')

# - Dropping Nan columns
data=data.dropna()

y=data['median_house_value'].values
X=data.drop(['median_house_value','ocean_proximity'],axis=1).values

# - Splitting data into train and test
X_test,X_train,y_test,y_train=train_test_split(X,y,test_size=0.2,random_state=2)

# - Standardizing Data
m1=X_test.mean(axis=0)
std1=X_test.std(axis=0)
X_test=(X_test-m1)/std1
X_train=(X_train-m1)/std1


# -> Part 1 Solution
class regressor1:
    def __init__ (self,learningrate=0.01,epochs=100):
        self.intercept_=None
        self.coef_=None
        self.lr=learningrate
        self.epochs=epochs
        self.cost_history1=[]
    def dot(self,a,b):
        p = b.shape[1] 
        C = np.zeros((a.shape[0], p))
        for i in range(a.shape[0]):
            for j in range(p):
                for k in range(a.shape[1]):
                    C[i,j]+=a[i,k]*b[k,j]
        return C
    def mean(self, a):
        flat = a.reshape(-1)
        total = 0.0
        for val in flat:
            total += val
        return total / flat.size if flat.size else 0.0
      
    def cost_function(self,X_train,y_train):
        y_hat=self.dot(X_train,self.coef_.reshape(-1,1))+self.intercept_
        return self.mean((y_train-y_hat)**2)
        
    def fit(self,X_train,y_train):
        self.intercept_=0
        self.coef_=np.ones(X_train.shape[1])
        for k in range(self.epochs):
            y_hat=self.dot(X_train,self.coef_.reshape(-1,1))+self.intercept_
            error = y_train - y_hat  
            grad_b = -2 * self.mean(error)
            grad_w = np.zeros(X_train.shape[1])
            for j in range(X_train.shape[1]):
                for i in range(X_train.shape[0]):
                    grad_w[j] += -2 * error[i, 0] * X_train[i, j]
            grad_w /= X_train.shape[0]

            self.intercept_ -= self.lr * grad_b
            self.coef_ -= self.lr * grad_w
            cost = self.cost_function(X_train, y_train)
            self.cost_history1.append(cost)

    def predict(self,X_test):
        return self.dot(X_test, self.coef_.reshape(-1, 1)) + self.intercept_
    
lr1=regressor1(0.1,30)

start1=time.time()
lr1.fit(X_train,y_train)
ctime_part1=time.time()-start1

y_test_pred1=lr1.predict(X_test)
y_train_pred1=lr1.predict(X_train)


# -> Part 2 Solution
class regressor2:
    def __init__ (self,learningrate=0.01,epochs=100):
        self.intercept_=None
        self.coef_=None
        self.lr=learningrate
        self.epochs=epochs
        self.cost_history2=[]
    def cost_function(self,X_train,y_train):
        y_hat=X_train.dot(self.coef_)+self.intercept_
        return np.mean((y_train-y_hat)**2)
    def fit(self,X_train,y_train):
        self.intercept_=0
        self.coef_=np.ones(X_train.shape[1])
        for i in range(self.epochs):
            y_hat=X_train.dot(self.coef_)+self.intercept_
            s1=-2*np.mean(y_train-y_hat)
            s2=-2*(y_train-y_hat).dot(X_train)/X_train.shape[0]
            self.intercept_=self.intercept_-self.lr*s1
            self.coef_=self.coef_-self.lr*s2
            cost = self.cost_function(X_train, y_train)
            self.cost_history2.append(cost)
    def predict(self,X_test):
        return self.intercept_+X_test.dot(self.coef_)
    
lr2=regressor2(0.2,50)

start2=time.time()
lr2.fit(X_train, y_train)
ctime_part2=time.time()-start2

y_test_pred2=lr2.predict(X_test)
y_train_pred2=lr2.predict(X_train)


# -> Part 3 Solution
lr3=LinearRegression()

start3=time.time()
lr3.fit(X_train,y_train)
ctime_part3=time.time()-start3

y_test_pred3=lr3.predict(X_test)
y_train_pred3=lr3.predict(X_train)


# -> Part 1 Regression Metrics and Convergence Time
mae_train1 = mean_absolute_error(y_train, y_train_pred1)
rmse_train1 = np.sqrt(mean_squared_error(y_train, y_train_pred1))
r2_train1 = r2_score(y_train, y_train_pred1)

mae_test1 = mean_absolute_error(y_test, y_test_pred1)
rmse_test1 = np.sqrt(mean_squared_error(y_test, y_test_pred1))
r2_test1 = r2_score(y_test, y_test_pred1)

print("TRAINING SET part1 :")
print("MAE:", mae_train1)
print("RMSE:", rmse_train1)
print("R² Score:", r2_train1)

print("\nTEST/VALIDATION SET part1 :")
print("MAE:", mae_test1)
print("RMSE:", rmse_test1)
print("R² Score:", r2_test1)

print('\nConvergence time part2 : ',ctime_part1)


# -> Part 2 Regression Metrics and Convergence Time
mae_train2 = mean_absolute_error(y_train, y_train_pred2)
rmse_train2 = np.sqrt(mean_squared_error(y_train, y_train_pred2))
r2_train2 = r2_score(y_train, y_train_pred2)

mae_test2 = mean_absolute_error(y_test, y_test_pred2)
rmse_test2 = np.sqrt(mean_squared_error(y_test, y_test_pred2))
r2_test2 = r2_score(y_test, y_test_pred2)

print("TRAINING SET part2 :")
print("MAE:", mae_train2)
print("RMSE:", rmse_train2)
print("R² Score:", r2_train2)

print("\nTEST/VALIDATION SET part2 :")
print("MAE:", mae_test2)
print("RMSE:", rmse_test2)
print("R² Score:", r2_test2)

print('\nConvergence time part2 : ',ctime_part2)


# -> Part 3 Regression Metrics and Convergence Time
mae_train3 = mean_absolute_error(y_train, y_train_pred3)
rmse_train3 = np.sqrt(mean_squared_error(y_train, y_train_pred3))
r2_train3 = r2_score(y_train, y_train_pred3)

mae_test3 = mean_absolute_error(y_test, y_test_pred3)
rmse_test3 = np.sqrt(mean_squared_error(y_test, y_test_pred3))
r2_test3 = r2_score(y_test, y_test_pred3)

print("TRAINING SET part3 :")
print("MAE:", mae_train3)
print("RMSE:", rmse_train3)
print("R² Score:", r2_train3)

print("\nTEST/VALIDATION SET part3 :")
print("MAE:", mae_test3)
print("RMSE:", rmse_test3)
print("R² Score:", r2_test3)

print('\nFitting Duration part3 : ',ctime_part3)


# -> Plot of Cost Function Convergence
plt.plot(lr1.cost_history1,np.arange(30),label='part1')
plt.plot(lr2.cost_history2,np.arange(50),label='part2')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.legend()
plt.grid(True)
plt.show()

# ->Plot of Comparison of Regression Metrics
x_ = ['Part1', 'Part2', 'Part3']
x = np.arange(len(x_))
r2 = [r2_train1, r2_train2, r2_train3]
mae = [mae_train1, mae_train2, mae_train3]
rmse = [rmse_train1, rmse_train2, rmse_train3]

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 8))

ax[0].bar(x, mae, width=0.4, color='skyblue')
ax[0].set_ylabel('MAE')
ax[0].set_title('MAE Comparison')

ax[1].bar(x, rmse, width=0.4, color='orange')
ax[1].set_ylabel('RMSE')
ax[1].set_title('RMSE Comparison')

ax[2].bar(x, r2, width=0.4, color='green')
ax[2].set_ylabel('R²')
ax[2].set_title('R² Comparison')

ax[2].set_xlabel('Method')  

ax[0].set_xticks(x)
ax[0].set_xticklabels(x_)
ax[1].set_xticks(x)
ax[1].set_xticklabels(x_)
ax[2].set_xticks(x)
ax[2].set_xticklabels(x_)

plt.tight_layout()
plt.show()
