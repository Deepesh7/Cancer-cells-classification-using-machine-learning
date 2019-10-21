#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
data = pd.read_csv("wdbc.dataset",header=None)

#
## Dropping the id column from the dataset.
#
data=data.drop([0],axis=1)
data.head()


# In[2]:


column_names=[]
for i in range(1,32):
    column_names.append("Column"+str(i))

data.columns = column_names
data.head()


# In[3]:


#
## Changing the value of M = 1 and B = 0 in the dataset
#
data.Column1=pd.Series(np.where(data.Column1.values == 'M', 1, 0),data.index)
data.head()


# In[4]:


#
## Splitting the data into training set, validation set and test set.
#

data_train, data_test = train_test_split(data, train_size=0.8, test_size=0.2, random_state=7)
data_test,data_validate = train_test_split(data_test,train_size=0.5,test_size=0.5,random_state=7)

#
## Splitting vertically the label column and the rest of the features for all the set.
#
data_train_label = np.array(data_train.iloc[:,0:1])
data_train_x = np.array(data_train.iloc[:,1:])

data_test_label = np.array(data_test.iloc[:,0:1])
data_test_x = np.array(data_test.iloc[:,1:])

data_validate_label = np.array(data_validate.iloc[:,0:1])
data_validate_x = np.array(data_validate.iloc[:,1:])


# In[5]:


data_train.head()


# In[6]:


data_train.shape


# In[7]:


data_validate.head()


# In[8]:


data_validate.shape


# In[9]:


data_test.head()


# In[10]:


data_test.shape


# In[11]:


scaler = StandardScaler()
scaler.fit(data_train_x)

data_train_x = scaler.transform(data_train_x)
data_test_x = scaler.transform(data_test_x)
data_validate_x = scaler.transform(data_validate_x)


# In[12]:


def true_or_false(h):
    '''Converts Probability into 0 class or 1 class.
       If h>=0.5 then class 1 is returned else 
       if h<0.5 class 0 is returned'''
    
    h_true_or_false=[]
    for i in h:
        if(i[0] >= 0.5):
            h_true_or_false.append(1)
        elif(i[0] < 0.5):
            h_true_or_false.append(0)
    return h_true_or_false


# In[13]:


def loss_func(y,h):
    '''Computes the cost function'''
    j = ((y*(np.log(h)))+(1-y)*(np.log(1-h)))
    len_ = j.shape[0]
    j_theta = -np.sum(j)/len_
    return j_theta


# In[14]:


# h = Activation function/hypothesis
# data_train_label = y/ output labels
# data_train_x = attribute matrix of training data
def update_weights(h,data_train_label,data_train_x,weights_array,learning_rate,bias):
    ''' Calculates gradients and update weights and bias'''
    m = data_train_x.shape[1]
    
    derivate_weight = np.dot(np.transpose(data_train_x),(h-data_train_label)) / m
    derivate_bias = np.sum(h-data_train_label) / m
    weights_array = weights_array - learning_rate * derivate_weight
    bias = bias - learning_rate * derivate_bias
    
    return {"updated weights": weights_array,"updated bias": bias}


# In[15]:


#
## Hyperparameter tuning for different learning rates
#


learning_rates=[0.00012,0.0008,0.00005,0.00001]
validation_hyperparameters_result = []
training_hyperparameters_result = []
ctr=0
for lr in learning_rates:
    print(lr)
    weights_array = np.zeros((30,1),dtype=int)
    bias = 1
    losses=[]
    validation_result=[]
    accuracy_training_data=[]
    accuracy_validation_data=[]
    validation_hyperparameters_result.append([])
    training_hyperparameters_result.append([])
    for epoch in range(10000):
        data_train_x_transpose = np.transpose(data_train_x)
        weights_array_transpose=np.transpose(weights_array)
        data_validate_x_transpose = np.transpose(data_validate_x)
    
        z=np.matmul(weights_array_transpose,data_train_x_transpose)+bias
        h=1/(1+np.exp(-z))
        h = np.transpose(h)
        training_hyperparameters_result[ctr].append(loss_func(data_train_label,h))
    
        z1=np.matmul(weights_array_transpose,data_validate_x_transpose)+bias
        h1=1/(1+np.exp(-z1))
        h1 = np.transpose(h1)
        validation_hyperparameters_result[ctr].append(loss_func(data_validate_label,h1))

        updated_values = update_weights(h,data_train_label,data_train_x,weights_array,lr,bias)
        weights_array = updated_values["updated weights"]
        bias = updated_values["updated bias"]
        weights_array.shape
    print(validation_hyperparameters_result[ctr][0:5])
    ctr+=1


# In[16]:


plt.figure(figsize=(14,10))
plt.title("Plotting loss function for different learning rate on validation set")
c=0
for x in validation_hyperparameters_result:
    plt.plot(x,label = "Learning rate: "+str(learning_rates[c]))
    c+=1
plt.legend()


# ##### From the above graph we can conclude that 0.00012 best fits our data and thus it is choosen for our final model.

# In[17]:


plt.figure(figsize=(14,10))
plt.title("Plotting loss function for different learning rate on training set")
c=0
for y in training_hyperparameters_result:
    plt.plot(y, label = "Learning rate: "+str(learning_rates[c]))
    c+=1
plt.legend()


# #### By choosing the best learning rate through hyperparameter tuning we are training the logistic regression model and evaluating the performance on training and validation data through different evaluation metrics such as recall, precision and accuracy.

# In[18]:


weights_array = np.zeros((30,1),dtype=int)
bias = 1
learning_rate=0.00012
losses=[]
validation_result=[]
accuracy_training_data=[]
accuracy_validation_data=[]
for epoch in range(10000):
    data_train_x_transpose = np.transpose(data_train_x)
    weights_array_transpose=np.transpose(weights_array)
    data_validate_x_transpose = np.transpose(data_validate_x)
    
    z=np.matmul(weights_array_transpose,data_train_x_transpose)+bias
    h=1/(1+np.exp(-z))
    h = np.transpose(h)
    losses.append(loss_func(data_train_label,h))
    h_tof = true_or_false(h)
    accuracy_training_data.append(accuracy_score(data_train_label,h_tof))
    
    
    
    z1=np.matmul(weights_array_transpose,data_validate_x_transpose)+bias
    h1=1/(1+np.exp(-z1))
    h1 = np.transpose(h1)
    validation_result.append(loss_func(data_validate_label,h1))
    h1_tof = true_or_false(h1)
    accuracy_validation_data.append(accuracy_score(data_validate_label,h1_tof))
    
    if(epoch % 500 == 0):
       print(losses[epoch])

    updated_values = update_weights(h,data_train_label,data_train_x,weights_array,learning_rate,bias)
    weights_array = updated_values["updated weights"]
    bias = updated_values["updated bias"]
    weights_array.shape


# In[19]:


# For Training data

weights_array_transpose = np.transpose(weights_array)
z2=np.matmul(weights_array_transpose,data_train_x_transpose)+bias
h2=1/(1+np.exp(-z2))
h2 = np.transpose(h2)
h2_tof = true_or_false(h2)

ps_train=precision_score(data_train_label, h2_tof)
print(ps_train)
recall_train = recall_score(data_train_label, h2_tof)
print(recall_train)
f1_train=f1_score(data_train_label, h2_tof)
print(f1_train)


# #### Precision, Recall and Accuracy score for Training data

# In[20]:


cr = classification_report(data_train_label,h2_tof,target_names=['Benign','Malignant'])
print("                 Training set    \n")
print(cr)
accuracy_training = accuracy_score(data_train_label,h2_tof)
print('\n\nAccuracy for training data: '+str(accuracy_training))


# In[21]:


# For Validation data

data_validate_x_transpose = np.transpose(data_validate_x)
z3=np.matmul(weights_array_transpose,data_validate_x_transpose)+bias
h3=1/(1+np.exp(-z3))
h3 = np.transpose(h3)
h3_tof = true_or_false(h3)

ps_validate=precision_score(data_validate_label, h3_tof)
print(ps_validate)
recall_validate = recall_score(data_validate_label, h3_tof)
print(recall_validate)
f1_validate=f1_score(data_validate_label, h3_tof)
print(f1_validate)


# #### Precision, Recall and Accuracy score for Validation data

# In[22]:


cr1 = classification_report(data_validate_label,h3_tof,target_names=['Benign','Malignant'])
print("                 Validation set    \n")
print(cr1)
accuracy_validate = accuracy_score(data_validate_label, h3_tof)
print("\n\nAccuracy for validation data: "+str(accuracy_validate))


# In[23]:


# For Test data

data_test_x_transpose = np.transpose(data_test_x)
z4=np.matmul(weights_array_transpose,data_test_x_transpose)+bias
h4=1/(1+np.exp(-z4))
h4 = np.transpose(h4)
h4_tof = true_or_false(h4)

ps_test=precision_score(data_test_label, h4_tof)
print(ps_test)
recall_test = recall_score(data_test_label, h4_tof)
print(recall_test)
f1_test=f1_score(data_test_label, h4_tof)
print(f1_test)


# #### Precision, Recall and Accuracy score for Test data

# In[24]:


cr2 = classification_report(data_test_label,h4_tof,target_names=['Benign','Malignant'])
print("                 Test set    \n")
print(cr2)
accuracy_test = accuracy_score(data_test_label,h4_tof)
print("\n\nAccuracy for test data: "+str(accuracy_test))


# #### Loss function for training and validation data. Since the curve for training data is greater than the curve for validation data, we can conclude that the model is not overfitting.

# In[25]:


plt.figure(figsize=(14,10))
plt.title("Loss function comparison for training and validation set")
plt.plot(losses, label = "loss function for training data")
plt.plot(validation_result, label = "loss function for validation data")
plt.legend()


# #### Accuracy score for validation and training data. We can see that the validation accuracy is more than the training accuracy, and this supports our claim that the model is not overfitting

# In[26]:


plt.figure(figsize=(14,10))
plt.title("Accuracy comparison for training and validation set")
plt.plot(accuracy_training_data, label = "Accuracy for training data")
plt.plot(accuracy_validation_data, label = "Accuracy for validation data")
plt.legend(loc="lower right")


# In[ ]:




