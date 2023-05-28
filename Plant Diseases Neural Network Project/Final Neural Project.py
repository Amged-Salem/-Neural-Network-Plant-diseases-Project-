#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display
from PIL import Image

# load the image file into a PIL image object
img = Image.open('dataset-cover.jpg')

# display the image in the notebook
display(img)


# # Import Libraries

# In[2]:


import os 
import glob as gb
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tqdm import tqdm
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay ,f1_score
import warnings
warnings.filterwarnings('ignore')


# # Load Dataset

# In[3]:


trainpath = "D:/Neural/Project/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/"
predpath  = "D:/Neural/Project/test/test/"
testpath = "D:/Neural/Project/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/"


# In[4]:


extension=[]
for folder in tqdm(os.listdir(trainpath)):
    for file in os.listdir(trainpath+"/"+folder+"/"):
        if os.path.isfile(trainpath+"/"+folder+"/"+file):
            extension.append ( os.path.splitext(file)[1])


# In[5]:


print(len(extension))


# In[6]:


for folder in tqdm(os.listdir(testpath)):
    for file in os.listdir(testpath+"/"+folder+"/"):
        if os.path.isfile(testpath+"/"+folder+"/"+file):
            extension.append ( os.path.splitext(file)[1])


# In[7]:


print(np.unique(extension))


# In[8]:


print(len(extension))


# In[9]:


for folder in tqdm(os.listdir(predpath)):
        if os.path.isfile(predpath+"/"+folder):
            extension.append ( os.path.splitext(folder)[1])


# In[10]:


print(np.unique(extension))


# In[11]:


print(len(extension))


# # Dimension of images

# In[12]:


size = []
count=[]
for folder in tqdm(os.listdir(trainpath)):
    files = gb.glob(pathname= str(trainpath + "/" + folder + "/*.JPG")) #+ \
#             gb.glob(os.path.join(trainpath, folder, '*.jpg'))
    for file in files :
        image = cv2.imread(file)
        size.append(image.shape)
        
pd.Series(size).value_counts()


# In[13]:


count.append(len(size))


# In[14]:


size = []
for folder in tqdm(os.listdir(testpath)):
    files = gb.glob(pathname= str(testpath + "/" + folder + "/*.JPG")) #+ \
#             gb.glob(os.path.join(testpath, folder, '*.jpg'))
    for file in files :
        image = cv2.imread(file)
        size.append(image.shape)
        
pd.Series(size).value_counts()


# In[15]:


count.append(len(size))


# In[16]:


count.append(33)


# In[17]:


count


# # Data Distribution

# In[18]:


labels=["Train","Test","Pred."]
# Create the pie chart
plt.pie(count, labels=labels, autopct='%1.0f%%')

# Add a title to the pie chart
plt.title('Data Distribution')

# Display the pie chart
plt.show()


# # Train Data Distribution

# In[19]:


values=[]
labels=[]
for folder in tqdm(os.listdir(trainpath)):
    files = gb.glob(pathname= str(trainpath + "/" + folder + "/*.JPG")) #+ \
#             gb.glob(os.path.join(trainpath, folder, '*.jpg'))
    print(f'for traing data ,{len(files)} in folder {folder}')
    values.append(len(files))
    labels.append(folder)


# In[20]:


plt.figure(figsize=(15,15))
plt.title('Train data distribution ',fontsize=30)
plt.xlabel('Number of image',fontsize=20)
plt.ylabel('Plant dieases',fontsize=20)

keys = list(labels)
# get values in the same order as keys, and parse percentage values
vals = list(values)
sns.barplot(y=keys, x=vals)


# # Test Data Distribution

# In[21]:


values=[]
labels=[]
for folder in tqdm(os.listdir(testpath)):
    files = gb.glob(pathname= str(testpath + "/" + folder + "/*.JPG")) #+ \
#             gb.glob(os.path.join(testpath, folder, '*.jpg'))
    print(f'for testing data ,{len(files)} in folder {folder}')
    values.append(len(files))
    labels.append(folder)


# In[22]:


plt.figure(figsize=(15,15))
plt.title('Test data distribution ',fontsize=30)
plt.xlabel('Number of image',fontsize=20)
plt.ylabel('Plant dieases',fontsize=20)

keys = list(labels)
# get values in the same order as keys, and parse percentage values
vals = list(values)
sns.barplot(y=keys, x=vals)


# # Pred. Data Distribution

# In[23]:


files = gb.glob(pathname= str(predpath + "/*.JPG"))#+ \
#             gb.glob(os.path.join(predpath, folder, '*.jpg'))
print(f'for pred. data ,{len(files)} image ')


# # Categories in Dataset

# In[24]:


lis=[]
for folder in tqdm(os.listdir(trainpath)):
    if folder not in lis :
        lis.append(folder)


# In[25]:


for folder in tqdm(os.listdir(testpath)):
    if folder not in lis :
        lis.append(folder)


# In[26]:


lis



# In[27]:


code = {key: value for value, key in enumerate(lis)}

print(code)


# In[28]:


def getcode(n) :    
    for x ,y in code.items() :
        if n == y :
            return x


# # New size of images 

# In[29]:


s=100


# # Reading and Visualization Train Data

# In[30]:


X_train=[]
Y_train=[]
for folder in tqdm(os.listdir(trainpath)):
    files = gb.glob(pathname= str(trainpath + "/" + folder + "/*.JPG")) #+ \
#             gb.glob(os.path.join(trainpath, folder, '*.jpg'))
    for file in files :
        image = cv2.imread(file)
        image_resized = cv2.resize(image,(s,s))
        X_train.append(list(image_resized))
        Y_train.append(code[folder])


# In[31]:


print(np.array(X_train).shape)
print(np.array(Y_train).shape)


# In[32]:


plt.figure(figsize=(30,30))
for n , i in enumerate(list(np.random.randint(0,len(X_train),36))):
    plt.subplot(6,6,n+1)
    plt.imshow(X_train[i])
    plt.axis('off')
    plt.title(getcode(Y_train[i]))


# # Reading and Visualization Test Data

# In[33]:


X_test=[]
Y_test=[]
for folder in tqdm(os.listdir(testpath)):
    files = gb.glob(pathname= str(testpath + "/" + folder + "/*.JPG")) #+ \
#             gb.glob(os.path.join(testpath, folder, '*.jpg'))
    for file in files :
        image = cv2.imread(file)
        image_resized = cv2.resize(image,(s,s))
        X_test.append(list(image_resized))
        Y_test.append(code[folder])


# In[34]:


print(np.array(X_test).shape)
print(np.array(Y_test).shape)


# In[35]:


plt.figure(figsize=(30,30))
for n , i in enumerate(list(np.random.randint(0,len(X_test),36))):
    plt.subplot(6,6,n+1)
    plt.imshow(X_test[i])
    plt.axis('off')
    plt.title(getcode(Y_test[i]))


# In[36]:


print(np.unique(Y_train))


# In[37]:


print(np.unique(Y_test))


# # Reading and Visualization Pred. Data

# In[38]:


X_pred = []
files = gb.glob(pathname= str(predpath  + "/*.JPG"))
for file in files :
    image = cv2.imread(file)
    image_resized = cv2.resize(image,(s,s))
    X_pred.append(list(image_resized))


# In[39]:


print(np.array(X_pred).shape)


# In[40]:


plt.figure(figsize=(30,30))
for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))):
    plt.subplot(6,6,n+1)
    plt.imshow(X_pred[i])
    plt.axis('off')


# # Preprocess the data

# In[41]:


X_train=np.array(X_train)
Y_train=np.array(Y_train)
X_test=np.array(X_test)
Y_test=np.array(Y_test)
X_pred=np.array(X_pred)


# In[42]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(X_pred.shape)


# In[43]:


X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_pred =X_pred.astype('float32') / 255.0


# In[44]:


Y_train = keras.utils.to_categorical(Y_train, 38)
Y_test = keras.utils.to_categorical(Y_test, 38)


# In[45]:


print(np.unique(Y_train))
print(np.unique(Y_test))


# # Building CNN Model

# In[52]:


model = keras.models.Sequential([
    keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(s,s,3)),
    keras.layers.MaxPool2D(4,4),
    keras.layers.Conv2D(64,kernel_size=(3,3),activation="relu"),
    keras.layers.MaxPool2D(4,4),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(38,activation="softmax"),
])


# In[53]:


print("Model Details :")
print(model.summary())


# # Compiling Model

# In[54]:


model.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss="categorical_crossentropy",metrics=["accuracy"])


# # Fitting Model

# In[55]:


history=model.fit(X_train,Y_train,epochs=10,batch_size=64,verbose=1)


# # Evaluating Model

# In[56]:


loss,acc=model.evaluate(X_test,Y_test)


# # Predicted Model

# In[57]:


pred = model.predict(X_test)


# In[58]:


print("Y_test shape: ", Y_test.shape)
print("pred shape: ", pred.shape)


# # Accuracy Measures

# In[59]:


print("Train Accuracy  : {:.2f} %".format(history.history['accuracy'][-1]*100))
print("Test Accuracy   : {:.2f} %".format(accuracy_score(Y_test.argmax(axis=1), pred.argmax(axis=1)) * 100))
print("Precision Score : {:.2f} %".format(precision_score(Y_test.argmax(axis=1), pred.argmax(axis=1), average='micro') * 100))
print("Recall Score    : {:.2f} %".format(recall_score(Y_test.argmax(axis=1), pred.argmax(axis=1), average='micro') * 100))
print("F1 Score        : {:.2f} %".format(f1_score(Y_test.argmax(axis=1), pred.argmax(axis=1), average='micro') * 100))


# In[60]:


plt.figure(figsize= (20,5))
cm = confusion_matrix(Y_test.argmax(axis=1), pred.argmax(axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=list(range(1,39)))
fig, ax = plt.subplots(figsize=(12,12))
disp.plot(ax=ax,colorbar= False,cmap = 'YlGnBu')
plt.title("Confusion Matrix")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# # Accuracy Plot

# In[61]:


plt.plot(history.history['accuracy'], label='acc')
#plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epochs')
plt.ylabel('Training acc')
plt.legend()
plt.show()
plt.savefig("Accuracy plot.jpg")


# # Loss Plot

# In[62]:


plt.plot(history.history['loss'], label='loss')
#plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.legend()
plt.show()
plt.savefig("Loss plot.jpg")


# # Saving Model

# In[63]:


model.save("D:/Neural/Project/New Plant Diseases Model.h5")


# # Predicting Data

# In[64]:


Y_pred = model.predict(X_pred)


# In[65]:


print(np.array(Y_pred).shape)


# In[66]:


Y_pred


# In[67]:


files = gb.glob(pathname= str(predpath + "/*.JPG"))#+ \
#             gb.glob(os.path.join(predpath, folder, '*.jpg'))
l=[]

for i in files :
    file_path = i
    split_path = str(file_path).split("\\")
    l.append(split_path[-1].split(".")[0])
print(l)


# In[68]:


labels=[]
for i in Y_pred :
    labels.append(getcode(np.argmax(i)))


# In[69]:


labels


# In[70]:


plt.figure(figsize=(30,30))
for i in range(33) :
    plt.subplot(6,6,i+1)
    plt.imshow(np.array(X_pred)[i])
    plt.axis('off')
    plt.title("Predicted : "+str(getcode(np.argmax(Y_pred[i])))+"\n"+"Actual : "+str(l[i]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




