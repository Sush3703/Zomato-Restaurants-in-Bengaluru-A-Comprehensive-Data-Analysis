#!/usr/bin/env python
# coding: utf-8

# ## Zomato Restaurent Data Analysis

# In[2]:


# importing requied liberaries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[3]:


#loading the data
df = pd.read_csv('zomato.csv')


# In[4]:


df.head()


# In[5]:


#Let's check the information about data
df.info()


# ## Overview of Data
# 
# 1. The data primarily focuses on Bangalore and its surrounding areas.
# 2. The restaurants can be categorized into various types based on their serving style, such as QSR (Quick Service Restaurants),    Buffets, Quick Bites, etc.
# 3. The dataset contains 17 columns with different information, some of which are crucial for analysis.
# 4. Some restaurants serve multiple cuisines, so a new category could be created for them.
# 5. One key detail missing from the data is whether a restaurant is purely vegetarian, which is an important factor considering    that India has a predominantly vegetarian population.
# 6. The "Cost for 2 people" column is an important variable in the dataset, as it serves as a dependent variable for analysis

# In[6]:


df.shape


# In[7]:


#first let drop the unncessary columns
df.drop(['url','address','menu_item', 'phone','reviews_list'],axis=1 , inplace=True)


# In[8]:


df.isnull().sum()/len(df)


# In[9]:


df.duplicated().sum()


# In[10]:


df.drop_duplicates(inplace=True)


# In[11]:


df.columns


# In[12]:


df["rate"].unique()


# In[13]:


def clean(text):
    text=str(text)
    text=re.sub(r"/+","",text) 
    text=re.sub("[^0.0-9.0]","",text)
    return text
#Here we defined a function to remove /5 and nonnumeric char


# In[14]:


df['rate'] = df['rate'].apply(clean)
df['rate'] = pd.to_numeric(df['rate'])
df['rate'].unique()


# In[15]:


# fill null values with mean
df['rate'].fillna(df['rate'].mode(), inplace =True)
df['rate'].unique()


# In[16]:


#Lets clean the name column  too
def clean(text):
    text=str(text)
    text=re.sub("[^a-zA-Z0-9]","",text) #Remove non alphanumeric character
    return text
df['name'] = df['name'].apply(clean)


# In[17]:


df.head(1)


# ## 1.Area wise Top 3 Restaurents according to rate

# In[18]:


df['listed_in(city)'].unique()


# In[19]:


#Let's make a pivot table which will show area wise top rated restaurents 
df =df.sort_values(by=['listed_in(city)', 'rate'], ascending = [True, False])#sorted vales by city and ratings
df['rank'] = df.groupby('listed_in(city)')['name'].rank(method='first', ascending=False)#ranked the restaurents by ratings
top_5_restaurants = df[df['rank'] <= 5]
top_5_restaurants = top_5_restaurants.pivot_table(
                                  index ='listed_in(city)',
                                  columns ='rank',
                                  values = 'name',
                                  aggfunc = 'first',
                                  dropna= True)


# In[20]:


top_5_restaurants


# ## 2.Leader in Franchise Business

# In[21]:


sns.countplot(data= df, x = 'name', width =0.5,
             order= df.name.value_counts().iloc[:10].index)
plt.xticks(rotation=90, ha='right', fontsize=10)
plt.show


# ## 3. Top 10 Expensive  and  Low-Priced Restaurent

# In[22]:


df['approx_cost(for two people)'].unique()


# In[23]:


df["approx_cost(for two people)"]=df["approx_cost(for two people)"].str.replace(",","")
#Replace the comma
df["approx_cost(for two people)"]=df["approx_cost(for two people)"].str.replace("nan","")
df["approx_cost(for two people)"]=pd.to_numeric(df["approx_cost(for two people)"])      #Convert to Numeric
df.rename(columns={"approx_cost(for two people)":"Cost2People"},inplace=True)           #Rename the column
df["Cost2People"].unique()


# In[24]:


df['Cost2People'].fillna(df['Cost2People'].mean() , inplace = True)


# In[25]:


df_sorted = (df.sort_values(by ='Cost2People', ascending = False)).head(10)

plt.figure(figsize = (10,6))
plt.bar(x = df_sorted['name'],
        height = df_sorted['Cost2People'],
        width = 0.5)
plt.xlabel('Restaurents')
plt.ylabel('Cost For 2 People')
plt.xticks(rotation=45, ha='right', fontsize=10)


# In[26]:


df_sorted = (df.sort_values(by ='Cost2People')).head(10)

plt.figure(figsize = (10,6))
plt.bar(x = df_sorted['name'],
        height = df_sorted['Cost2People'],
        width = 0.5)
plt.xlabel('Restaurents')
plt.ylabel('Cost For 2 People')
plt.xticks(rotation=45, ha='right', fontsize=10)


# ## 4.Favourite Cusine of Banglore

# In[27]:


cuisines1= df['cuisines'].value_counts().head(10)
X = cuisines1.index
Y = cuisines1.values
sns.barplot(x=Y,y=X)


# ## 5. Distribution of Online order availability

# In[28]:


Online_order = df['online_order'].value_counts()
Online_order.index
plt.pie(x =Online_order.values , labels = Online_order.index, autopct='%1.2f%%')


# ## 6.Avg cost for 2 people by location

# In[29]:


loc_cost = df.groupby('listed_in(city)')['Cost2People'].mean().reset_index()
loc_cost_ordered = loc_cost.sort_values(by="Cost2People", ascending=False)['listed_in(city)'].to_list()

sns.barplot(x=loc_cost['Cost2People'], y=loc_cost['listed_in(city)'],
            order=loc_cost_ordered)
plt.xticks(rotation=90)
plt.show()


# ## 7. High rated restaurents with high vote counts

# In[30]:


df.sort_values(by=['rate', 'votes'], ascending= False).reset_index().head(10)


# In[ ]:





# ##  8.Avg cost for 2 people if restaurent has table booking option or not

# In[490]:


od= df.groupby('book_table')['Cost2People'].mean()
plt.bar(x=od.index, height=od.values)


# ## 9.Correlation between the table which have numeric values in the table

# In[506]:


cr=df.corr(numeric_only=True)
sns.heatmap(cr,annot=True)


# ## 10.Top 10 loved cuisines of Banglore

# In[31]:


dl = df['dish_liked'].value_counts().reset_index().head(10)
plt.barh(dl['dish_liked'], dl['count'])


# # # Inferences
# 1. By analyzing the pivot table, we can identify industry leaders in specific areas. This information can help new          entrepreneurs develop strategies when opening a restaurant in those regions.
# 2. Since several restaurant names appear repeatedly, it suggests they may follow a franchise model.
# 3. A quick glance at the most expensive and least expensive restaurants shows a wide price variation in Bangalore. The least    expensive restaurant serves two people for as low as Rs 40, while the priciest ones can cost up to Rs 5000 for two.
# 4. Although Bangalore is a South Indian city, the most popular cuisine in the area is North Indian, as seen from the data on  popular cuisines.
# 5. Around 58% of restaurants offer online ordering facilities, while the rest do not, likely due to the smaller size of some establishments.
# 6. Areas like Church Street, Brigade Road, MG Road, Lavelle Road, and Residency Road have higher purchasing power compared to other parts of the city.
# 7. Looking at the list of top-rated restaurants based on the number of reviews, it's clear that only three restaurant chains dominate the rankings.
# 8. There is a significant difference in the average cost for two people depending on whether table booking is available. This disparity is due to QSR restaurants, which typically do not offer table reservations.
# 9. Biryani is the most popular cuisine in Bangalore, as it appears frequently across various restaurants in the city.
# 

# In[ ]:




