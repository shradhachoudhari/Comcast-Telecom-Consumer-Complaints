#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("/Users/shraddhalipane/Downloads/Comcast_telecom_complaints_data.csv")    #load csv dataset


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df['Date']=pd.to_datetime(df['Date'])        # number of complaints on daily basis


# In[6]:


df.head()


# In[7]:


df['month']=df['Date'].dt.month_name()      # created column 'month' 


# In[8]:


df


# In[9]:


df.groupby('month').size()      


# In[10]:


month=df.groupby('month').size()


# In[11]:


df.groupby('month').size()     


# In[12]:


month_df=pd.DataFrame(month).reset_index()


# In[13]:


month_df                                    # number of complaints on Monthly basis


# In[14]:


month_df.plot(x='month',y=0,kind='bar')
plt.show()


# In[15]:


by_date=df.groupby('Date').size()


# In[16]:


daily_df=pd.DataFrame(by_date).reset_index()


# In[17]:


daily_df.rename(columns={0:'total_cmp'},inplace=True)


# In[18]:


daily_df.plot(x='Date',y='total_cmp')                        # number of complaints on daily basis using plot


# In[19]:


daily_df.sort_values(by='total_cmp', ascending=False)


# In[85]:


df['Customer Complaint'] = df['Customer Complaint'].str.title() 
frequency = df['Customer Complaint'].value_counts()                    #Provide a table with the frequency of complaint types


# In[86]:


frequency


# In[22]:


import nltk


# In[23]:


nltk.download ()


# In[24]:


get_ipython().system('pip install wordcloud')


# In[25]:


from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


# In[26]:


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join([ch for ch in stop_free if ch not in exclude])
    normalised = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalised


# In[27]:


doc_complete = df["Customer Complaint"].tolist()
frequency= [clean(doc).split() for doc in doc_complete]


# In[28]:


conda install -c conda-forge gensim


# In[59]:


conda install jupyter


# In[60]:


import gensim
from gensim import corpora


# In[61]:


dictionary = corpora.Dictionary(frequency)
print(dictionary)


# In[66]:


doc_term_matrix = [dictionary.doc2bow(doc) for doc in frequency]
doc_term_matrix


# In[67]:


from gensim.models import LdaModel


# In[68]:


Num_Topic = 9
ldamodel = LdaModel(doc_term_matrix, num_topics= Num_Topic, id2word= dictionary, passes= 30)


# In[69]:


topics = ldamodel.show_topics()
for topic in topics:
    print(topic)
    print()


# In[70]:


word_dict = {}
for i in range(Num_Topic):
    words = ldamodel.show_topic(i, topn =20)
    word_dict["topic # " + "{}".format(i)] = [i[0] for i in words]


# In[71]:


pd.DataFrame(word_dict)


# In[72]:


df['Status'].unique()


# In[73]:


df['new_Status']=['Open' if st=="Open" or st=="Pending" else "Closed" for st in df['Status']]         #4.Created new categorical variable with value as Open and Closed 


# In[74]:


df


# In[75]:


df.groupby(['State','new_Status']).size().unstack()


# In[76]:


state_complain=df.groupby(['State','new_Status']).size().unstack()


# In[77]:


state_complain.plot.bar(stacked=True,figsize=(10,7))                    #Provide state wise status of complaints in a stacked bar chart. 


# In[78]:


df.groupby('State').size()


# In[79]:


df.groupby('State').size().sort_values(ascending=False)       


# In[80]:


len(df.groupby('State').size().sort_values(ascending=False))


# In[81]:


df.groupby('State').size().sort_values(ascending=False)[0:5]    #create bar graph


# In[82]:


df.groupby(['State','new_Status']).size().unstack()


# In[83]:


df.groupby(['State','new_Status']).size().unstack().fillna(0).sort_values(by='Open',ascending=False)


# In[84]:


unresolved_data=df.groupby(['State','new_Status']).size().unstack().fillna(0).sort_values(by='Open',ascending=False)


# In[51]:


unresolved_data


# In[52]:


unresolved_data['unresolved_cmp_percentage']=unresolved_data['Open']/unresolved_data['Open'].sum()*100            #unresolved complaints 


# In[53]:


unresolved_data


# In[54]:


df.groupby(['Received Via','new_Status']).size().unstack()


# In[55]:


resolved_data=df.groupby(['Received Via','new_Status']).size().unstack()


# In[56]:


resolved_data


# In[57]:


resolved_data['resolved']=resolved_data['Closed']/resolved_data['Closed'].sum()*100


# In[58]:


resolved_data['resolved']


# In[ ]:




