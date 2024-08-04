#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load the necessary python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

original = pd.read_csv("Merged file.csv")
#Load the dataset
df = original[original["Training count"].notnull()]


# In[21]:


#Print the first 5 rows of the dataframe.
df.head()


# In[3]:


# observe the shape of the dataframe.
df.shape


# In[5]:


#Let's create numpy arrays for features and target
X = df.drop('Output',axis=1).values
X = df.drop('Training count',axis=1).values
X = df.drop('Count',axis=1).values
y = df['Output']
y.unique()


# In[6]:


y


# In[7]:


#importing train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y, shuffle=True)


# In[19]:


X_train.shape


# In[20]:


X_test.shape


# In[12]:


#import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
#setting up 
for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    
      #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 


# In[13]:


#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[14]:


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=5)


# In[15]:


#Fit the model
knn.fit(X_train,y_train)


# In[17]:


#Get accuracy. Note: In case of classification algorithms score method represents accuracy.
knn.score(X_test,y_test)


# In[ ]:




