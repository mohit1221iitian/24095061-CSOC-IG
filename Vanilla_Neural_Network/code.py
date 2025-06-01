#!/usr/bin/env python
# coding: utf-8

# In[119]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 
import tracemalloc
import seaborn as sns


# #  Data Pre Processing

# In[120]:


data=pd.read_csv('KaggleV2-May-2016.csv')
data.head(1)


# In[121]:


data.drop(['PatientId','AppointmentID'],axis=1,inplace=True)
data['No-show']=data['No-show'].map({'No':0,'Yes':1})
data['Gender']=data['Gender'].map({'F':0,'M':1})

data=pd.get_dummies(data,columns=['Neighbourhood'],drop_first=True)

data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay'])
data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay'])
data['DaysBetween'] = (data['AppointmentDay'].dt.date - data['ScheduledDay'].dt.date).apply(lambda x: x.days)
data.drop(['AppointmentDay','ScheduledDay'],axis=1,inplace=True)

data=data.astype(int)

data = data[data['Age'] >= 0]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
data[['Age', 'DaysBetween']] = scaler.fit_transform(data[['Age', 'DaysBetween']])


# In[122]:


data.shape


# In[123]:


data.value_counts('No-show')


# In[124]:


y = data['No-show']
X = data.drop(['No-show'], axis=1)


# In[125]:


from sklearn.model_selection import train_test_split


# In[126]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# # tensorflow-keras implementation

# In[206]:


import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
import keras_tuner as kt
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix,f1_score,accuracy_score, precision_recall_curve, auc


# In[128]:


from sklearn.utils import class_weight
import numpy as np
y_train1 = np.array(y_train).ravel()
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train1),
    y=y_train1
)
class_weights = dict(zip(np.unique(y_train_array), weights))


# In[129]:


pos_weight = class_weights[1] / class_weights[0]


# In[130]:


from tensorflow.keras import backend as K

def weighted_binary_crossentropy(pos_weight):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        log_loss = -(pos_weight * y_true * K.log(y_pred) +
                     (1 - y_true) * K.log(1 - y_pred))
        return K.mean(log_loss)
    return loss


# In[139]:


def build_model(hp):
    model=Sequential()
    model.add(Input(shape=(X.shape[1],)))

    for i in range(hp.Int('layers',min_value=1,max_value=10)):
            model.add(Dense(hp.Int('units'+str(i),min_value=32,max_value=256,step=32),
                            activation='relu',
                            kernel_regularizer=regularizers.l2(hp.Choice('l2'+str(i),values=[1e-4, 1e-3, 1e-2]))))
            model.add(Dropout(hp.Choice('d'+str(i),values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
    model.add(Dense(1,activation='sigmoid'))

    hp_lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_lr),
                  loss=weighted_binary_crossentropy(pos_weight),
                  metrics=['accuracy'])
    return model


# In[140]:


tuner=kt.RandomSearch(build_model,
                     objective='val_accuracy',
                     max_trials=3,
                     directory='firstcsoc',
                     project_name='9')


# In[141]:


tuner.search(X_train,y_train,epochs=5,validation_data=(X_test,y_test),batch_size=32,class_weight=class_weights)


# In[142]:


callback=EarlyStopping(monitor='val_loss',
                       min_delta=0.001,
                       patience=3,
                       verbose=0,
                       mode='auto',
                       baseline=None,
                       restore_best_weights=True)


# In[143]:


model=tuner.get_best_models(num_models=1)[0]


# In[144]:


model.summary()


# In[145]:


tracemalloc.start()
start_Ttime=time.time()

history=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=75,batch_size=32,callbacks=callback,initial_epoch=6,class_weight=class_weights)

end_time=time.time()
current,peak=tracemalloc.get_traced_memory()
tracemalloc.stop()


# # Model evaluation and analysis (using tensorflow-keras implementation) 

# In[163]:


plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()


# In[207]:


y_pred = (pred_probs > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
f1_macro= f1_score(y_test, y_pred,average='macro')
f1_weighted= f1_score(y_test, y_pred,average='weighted')
f1_s= f1_score(y_test, y_pred)

precision, recall, _ = precision_recall_curve(y_test, y_pred)
pr_auc = auc(recall, precision)


# In[208]:


conf_matrix = confusion_matrix(y_test, y_final_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[210]:


print(f"Current memory usage: {current / 1e6:.2f} MB; Peak: {peak / 1e6:.2f} MB")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (macro) : {f1_macro:.4f}")
print(f"F1 Score (weighted) : {f1_weighted:.4f}")
print(f"F1 Score : {f1_s:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Training time: {end_time - start_Ttime:.2f} seconds")


# # Neural Network from scratch

# In[191]:


def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for i in range(1, L):
        parameters['w' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(2 / layer_dims[i - 1])
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

    return parameters


# In[192]:


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def leaky_relu(Z, alpha=0.01):
    return np.where(Z > 0, Z, alpha * Z)

def leaky_relu_derivative(Z, alpha=0.01):
    return np.where(Z > 0, 1, alpha)


# In[197]:


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    total = y.shape[1]
    weights = {cls: total / (2 * count) for cls, count in zip(classes, counts)}
    return weights


# In[193]:


def forward_propagation(X, parameters):
    W1, b1 = parameters['w1'], parameters['b1']
    W2, b2 = parameters['w2'], parameters['b2']
    
    Z1 = np.dot(W1, X) + b1
    A1 = leaky_relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = (X, Z1, A1, Z2, A2)
    return A2, cache


# In[194]:


def compute_loss(y, y_hat, weights):
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    w1 = weights[1]
    w0 = weights[0]
    loss = -np.mean(w1 * y * np.log(y_hat) + w0 * (1 - y) * np.log(1 - y_hat))
    return loss


# In[200]:


def backward_propagation(parameters, cache, y, weights):
    X, Z1, A1, Z2, A2 = cache
    m = X.shape[1]

    w1 = weights[1]
    w0 = weights[0]

    dZ2 = (A2 - y) * (w1 * y + w0 * (1 - y))  
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dA1 = np.dot(parameters['w2'].T, dZ2)
    dZ1 = dA1 * leaky_relu_derivative(Z1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads


# In[196]:


def update_parameters(parameters, grads, learning_rate=0.01):
    parameters['w1'] -= learning_rate * grads['dW1']
    parameters['b1'] -= learning_rate * grads['db1']
    parameters['w2'] -= learning_rate * grads['dW2']
    parameters['b2'] -= learning_rate * grads['db2']
    return parameters


# In[198]:


def train_model(X, y, layer_dims, epochs=1000, learning_rate=0.01):
    parameters = initialize_parameters(layer_dims)
    losses = []

    weights = compute_class_weights(y)

    for i in range(epochs):
        y_hat, cache = forward_propagation(X, parameters)
        loss = compute_loss(y, y_hat, weights)
        grads = backward_propagation(parameters, cache, y, weights)
        parameters = update_parameters(parameters, grads, learning_rate)

        losses.append(loss)
        if i % 100 == 0 or i == epochs - 1:
            print(f"Epoch {i}, Loss: {loss:.6f}")

    return parameters, losses


# In[199]:


X = X_train.T
y = y_train.values.reshape(1, -1)
layer_dims = [X.shape[0], 10, 1]

tracemalloc.start()
start_time1=time.time()

parameters, losses = train_model(X, y, layer_dims, epochs=1000, learning_rate=0.1)

end_time1=time.time()
current1,peak1=tracemalloc.get_traced_memory()


# # Model evaluation and analysis (using pure pyton implementation) 

# In[202]:


plt.plot(losses,label='loss')
plt.legend()
plt.show()


# In[211]:


y_pred_val, _ = forward_propagation(X_test.T, parameters)
y_pred_labels = (y_pred_val > 0.5).astype(int).flatten()
y_true_labels = y_test.values.flatten()

accuracy1 = accuracy_score(y_true_labels, y_pred_labels)

f11 = f1_score(y_true_labels, y_pred_labels,average='macro')
f12 = f1_score(y_true_labels, y_pred_labels,average='weighted')


precision1, recall1, _ = precision_recall_curve(y_true_labels, y_pred_val.flatten())
pr_auc1 = auc(recall, precision)


# In[212]:


conf_matrix = confusion_matrix(y_test, y_pred_labels)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[213]:


print(f"Current memory usage: {current1 / 1e6:.2f} MB; Peak: {peak1 / 1e6:.2f} MB")
print(f"Accuracy: {accuracy1:.4f}")
print(f"F1 Score (macro) : {f11:.4f}")
print(f"F1 Score (weighted) : {f12:.4f}")
print(f"PR-AUC: {pr_auc1:.4f}")
print(f"Training time: {end_time1 - start_time1:.2f} seconds")


# In[ ]:




