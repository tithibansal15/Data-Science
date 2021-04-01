#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import os
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import time


# In[2]:


indf = pd.read_excel('Input file_P2P_17mar21.xlsb',skiprows=3, engine='pyxlsb')
indf = indf.iloc[:,2:]
indf = indf[indf['Country Ultimate Street']!=0]

dfsub = indf[['Country Ultimate Name', 'Country Ultimate Street']]
dfsub = dfsub.dropna(how='any')
dfsub = dfsub.drop_duplicates()
dfsub = dfsub.apply(lambda x: x.astype(str).str.upper())

df_p23 = pd.read_excel('P23 Vendors Extract.xlsb', engine='pyxlsb')
df_p80 = pd.read_excel('P80 Vendors Extract.xlsb', engine='pyxlsb')

df_p23 = df_p23.iloc[:,0:3]
df_p80 = df_p80.iloc[:,0:3]

df_p23sub = df_p23.dropna(how='any')
df_p80sub = df_p80.dropna(how='any')

df_p23sub = df_p23sub.apply(lambda x: x.astype(str).str.upper())
df_p80sub = df_p80sub.apply(lambda x: x.astype(str).str.upper())


# In[14]:


df_p80sub['tst'] = [" ".join([i for i in str(x).split(" ") if (i.isalnum() and not i.isdigit())]) for x in df_p80sub['Street']]


# In[25]:


df_p80sub = df_p80sub[df_p80sub['tst']!=""]


# In[28]:


df_p23sub = df_p23sub.dropna(how='any')
df_p23sub = df_p23sub.drop_duplicates()

df_p80sub = df_p80sub.dropna(how='any')
df_p80sub = df_p80sub.drop_duplicates()


# In[29]:


df_p80sub.loc[:,'Name-Add'] = df_p80sub['Name 1']+" - "+df_p80sub['Street']
dfsub.loc[:,'Name-Add'] = dfsub['Country Ultimate Name'] + " - " + dfsub['Country Ultimate Street']


# In[5]:


df_p23sub.loc[:,'Name-Add'] = df_p23sub['Name']+" - "+df_p23sub['Street']
dfsub.loc[:,'Name-Add'] = dfsub['Country Ultimate Name'] + " - " + dfsub['Country Ultimate Street']


# In[31]:


choices = df_p80sub['Name-Add'].to_list()


# In[32]:


df_p80sub.shape


# In[ ]:


from datetime import datetime
start = datetime.now()
print(start)
for idx in range(0,10000,100):
    print(idx)
    sub = dfsub[idx:idx+10]
    sub.loc[:,'p80'] = [process.extract(x, choices, limit=5) for x in sub['Name-Add']]
    sub.to_csv('p2p_mapping_p80_5matches'+str(idx)+'-'+str(idx+100)+'.csv',index=False)
    end = datetime.now()
    print(end)
runtime = start-end
print("-----Duration-----",runtime)


# In[ ]:




