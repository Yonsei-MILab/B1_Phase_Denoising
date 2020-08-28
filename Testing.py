#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from scipy import io
import matplotlib.pyplot as plt
import numpy as np


# In[1]:


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used


# In[ ]:


# Load in-vivo test data
# test_data = io.loadmat('training_Dataset.mat')
test = np.array(test_data['test_avg_1'])


# In[ ]:


# Trained weight upload
#model.load_weights('./Phase_denoising_Weight.h5')


# In[ ]:


# Result
resY = model.predict(test)

io.savemat('./Phase_denoising_result.mat',{'X_train':test,'resY':resY})

