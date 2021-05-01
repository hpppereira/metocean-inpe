#!/usr/bin/env python
# coding: utf-8

# ## Processamento dos dados da boia Spotter na Antartica

# In[57]:


import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[61]:


# df = pd.read_csv('data/0130_SPC.CSV', header=0)
flt = pd.read_csv('data/0130_FLT.CSV', sep=',', index_col=False)
flt


# In[63]:


for c in flt.columns:
    plt.figure()
    flt[c].plot()


# In[59]:


flt['GPS_Epoch_Time(s)'].plot()


# In[46]:





# In[51]:


a = pd.to_datetime([datetime.fromtimestamp(flt['GPS_Epoch_Time(s)'][i]) for i in range(len(flt))])


# In[65]:


dt = a[4] - a[3]


# In[67]:


1/dt.total_seconds()


# In[68]:


a


# In[60]:


plt.plot(a)

