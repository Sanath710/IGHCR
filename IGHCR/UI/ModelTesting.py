
# coding: utf-8

# In[78]:


import os, cv2
import numpy as np, matplotlib.pyplot as plt
from keras.models import load_model


# In[319]:


consonants = {'1 k': "ક", '10 tha': "ઠ", '11 da': "ડ", '12 ddha': "ઢ", '14 ta': "ત", '15 tha': "થ", '16 da': "દ",'8 jha': "ઝ", 
              '17 dha': "ધ", '18 na': "ન", '19 pa': "પ", '2 kha': "ખ", '20 pha': "ફ", '21 ba': "બ",  '22 bha': "ભ", 
              '23 ma': "મ", '24 ya': "ય", '25 ra': "ર", '27 va': "વ",  '29 SA': "ષ", '3 ga': "ગ", '30 sa': "સ", '9 ta': "ટ",
              '32 ala': "ળ", '33 ksh': "ક્ષ", '34 jna': "જ્ઞ", '4 gha': "ઘ", '5 ca': "ચ", '6 cha': "છ", '7 ja': "જ", 30:'?'}


# In[179]:


def loadClass() :
    d = {'1 k': 0, '10 tha': 1, '11 da': 2, '12 ddha': 3, '14 ta': 4, '15 tha': 5, '16 da': 6, '17 dha': 7, '18 na': 8,
         '19 pa': 9, '2 kha': 10, '20 pha': 11, '21 ba': 12, '22 bha': 13, '23 ma': 14, '24 ya': 15, '25 ra': 16, '27 va': 17, 
         '29 SA': 18, '3 ga': 19, '30 sa': 20, '32 ala': 21, '33 ksh': 22, '34 jna': 23, '4 gha': 24, '5 ca': 25, '6 cha': 26,
         '7 ja': 27, '8 jha': 28, '9 ta': 29}
    return {v:k for k, v in d.items()}


# In[31]:


def loadModel(dir_path, model_name) : 
    return load_model(os.path.join(dir_path, model_name))


# In[32]:


#model = loadModel("Models","model-accur-[89.80-BETTER].h5")
model = loadModel("Models","model-accur-[94.19 - BEST].h5")


# In[33]:


model.summary()


# In[77]:


def loadImg(inp_img) :
    return cv2.imread(inp_img)


# In[149]:


def preprocessImg(inp_img) :
    resize = cv2.resize(inp_img, (64, 64))
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return gray, binary, np.array(cv2.ximgproc.thinning(binary,  thinningType=cv2.ximgproc.THINNING_GUOHALL))


# In[265]:


def classifyImg(img_arr, model) :
    img_arr = preprocessImg(img_arr)[2]
    img = np.divide(img_arr, 255).reshape(-1, 64, 64, 1)
    prediction = model.predict(img)
    return loadClass()[np.argmax(prediction)], round(np.max(prediction), 4) 


# In[317]:


# img = loadImg("test-k.png")
# preprocess = preprocessImg(img)
# classifyImg(preprocess, model)

