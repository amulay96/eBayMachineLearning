#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
df_attributes = pd.read_csv('mlchallenge_set.tsv',encoding='utf8',sep='\t', nrows =15000 , usecols=([5]))
print(df_attributes)


# In[2]:


for i in range(len(df_attributes)):
    df_attributes['attributes'][i] = df_attributes['attributes'][i][1:-1]
    df_attributes['attributes'] = df_attributes['attributes']
for i in range(len(df_attributes['attributes'])):
    df_attributes['attributes'][i] = re.split(":+", df_attributes['attributes'][i])


# In[3]:


def comma(x):
    for i in range(len(x)):
        x[i] = x[i].split(',')
    return x
for i in range(len(df_attributes['attributes'])):
    df_attributes['attributes'][i] = comma(df_attributes['attributes'][i])


# In[4]:


def to_dictionary(x):
    prop = []
    val = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            if j == len(x[i])-1:
                prop.append(x[i][j])
    prop = prop[:-1]
    
    val = x[1:]
    for i in range(len(val)-1):
        val[i] = val[i][:-1]    
    
    return dict(zip(prop, val))
for i in range(len(df_attributes['attributes'])):
    df_attributes['attributes'][i] = to_dictionary(df_attributes['attributes'][i])


# In[5]:


keys = []             # taking all keys for row of dictionary
for i in df_attributes['attributes'].index.values:
    keys.extend(list(df_attributes['attributes'][i].keys()))
list(set(keys))#Checking all unique keys for rows of dictionary


# In[6]:


len(list(set(keys)))
df = df_attributes['attributes']
df
lb = [] # check unique values of brand
for i in df.index.values:
    l = list(df[i].keys())
    if 'Brand' in l:
        if df[i]['Brand'] in lb:
            continue
        else:
            lb.append(df[i]['Brand'])
print(lb)
len(lb)


# In[7]:


l=[]
c=0
for i in df.index.values:
    l = df[i].keys()
    if "Brand" in l:
        c=c+1
        #print(dfn[i])
print(c)


# # creating the dataframe with shortlisted parameters
# 

# In[8]:


data = pd.DataFrame(np.nan, index= list(range(0,15000)), columns=['Brand', 'Product Line', 'Style', 'Color'])


# In[9]:


for i in range(len(df)):
    if 'Brand' in df[i].keys():
        data['Brand'][i] = df[i]['Brand']
    if 'Product Line' in df[i].keys():
        data['Product Line'][i] = df[i]['Product Line']
    if 'Style' in df[i].keys():
        data['Style'][i] = df[i]['Style']
    if 'Color' in df[i].keys():
        data['Color'][i] = df[i]['Color']
   
data.dropna(inplace=True)


# In[10]:


data.reset_index(drop= True, inplace=True)
data


# In[11]:


for i in range(len(data)):
    print(data['Color'][i])


# # Cleaning Brand Column

# In[12]:


for i in range(len(data)):
    ls = []
    if len(data['Brand'][i])>1:
        ls.append(data['Brand'][i][0])
        data['Brand'][i] = ls


# In[13]:


for i in range(len(data)):
    data["Brand"][i] = data["Brand"][i][0]
    
for i in range(len(data)):
    print(data["Brand"][i])


# In[14]:


len(data["Brand"].unique())
for i in range(len(data)):
    data["Brand"][i] = data["Brand"][i].lower()
    
data['Brand'] = pd.factorize(data.Brand)[0]
data


# In[15]:


data.to_csv("data_2.csv")


# # Cleaning Product Line 

# In[16]:


c= 0
for i in range(len(data["Product Line"])):
    if len(data["Product Line"][i]) > 1:
        print(len((data["Product Line"][i])))
        
for i in range(len(data)):
    lst = []
    if len(data["Product Line"][i]) > 1:
        st = data["Product Line"][i][0]
        lst.append(st)
        data["Product Line"][i] = lst


# In[17]:


for i in range(len(data['Product Line'])):
    print(data['Product Line'][i])
for i in range(len(data)):#Making list to string
    data["Product Line"][i] = data["Product Line"][i][0]

for i in range(len(data)):#Lower case 
    data["Product Line"][i] = data["Product Line"][i].lower()
len(data['Product Line'].unique())


# In[18]:


data.rename(columns={"Product Line": "ProductLine"},inplace=True)
data['ProductLine'] = pd.factorize(data.ProductLine)[0]


# In[19]:


data.to_csv("data_3.csv")


# # cleaning style column

# In[20]:


for i in range(len(data["Style"])):
    print(data["Style"][i])
    
for i in range(len(data["Style"])):
    if len(data["Style"][i]) > 1:
        print((data["Style"][i]))
        
for i in range(len(data)):
    lst = []
    if len(data["Style"][i]) > 1:
        st = data["Style"][i][0]
        lst.append(st)
        data["Style"][i] = lst
        
for i in range(len(data['Style'])):
    print(data['Style'][i])
    
for i in range(len(data)):#Making list to string
    data["Style"][i] = data["Style"][i][0]
    
for i in range(len(data)):#Lower case 
    data["Style"][i] = data["Style"][i].lower()


# In[21]:


len(data['Style'].unique())
data['Style'] = pd.factorize(data.Style)[0]
data


# # cleaning color column 

# In[22]:


data['Color']
for i in range(len(data["Color"])):
    if data["Color"][i] == []:
        (data["Color"][i])="Blue"
for i in range(len(data["Color"])):
    print(data["Color"][i])
    
for i in range(len(data)):
    lst = []
    if len(data["Color"][i]) > 1:
        st = data["Color"][i][0]
        lst.append(st)
        data["Color"][i] = lst
for i in range(len(data)):#Making list to string
    data["Color"][i] = data["Color"][i][0]


# In[23]:


len(data['Color'].unique())
data['Color'] = pd.factorize(data.Color)[0]
data
data.to_csv("final_data.csv")


# In[24]:


data


# In[38]:


from sklearn.cluster import AffinityPropagation
clustering = AffinityPropagation().fit(data)
print(clustering)


# In[ ]:





# In[ ]:





# In[42]:


re=clustering.predict(data)


# In[26]:


le=len(clustering.cluster_centers_)
cluster_centers_indices = clustering.cluster_centers_indices_
labels = clustering.labels_
n_clusters_ = len(cluster_centers_indices)
print(n_clusters_)


# In[27]:


from sklearn.cluster import KMeans

km = KMeans(n_clusters=le, random_state=1)
new = data._get_numeric_data()
km.fit(new)
predict=km.predict(new)
len(predict)


# In[28]:


import matplotlib.pyplot as plt
plt.plot(predict)


# In[29]:


import seaborn as sns


# In[30]:


sns.scatterplot(data=data)


# In[31]:


p = np.arange(0,6768)
p


# In[37]:


from itertools import cycle
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = data[cluster_centers_indices[k]]
    plt.plot(data[class_members, 0], data[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.show()


# In[46]:


plt.scatter(p, re, c=p, cmap='coolwarm', alpha=0.25)


# In[ ]:




