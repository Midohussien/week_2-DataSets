#!/usr/bin/env python
# coding: utf-8

# # 1. Importing needed libraries:

# In[1]:


# importing libraries:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # 2.Reading files:

# In[2]:


# Read the datasets:
cabData = pd.read_csv('Cab_Data.csv')
cabData.head()


# In[3]:


cityData = pd.read_csv('City.csv')
cityData.head()


# In[4]:


customerData = pd.read_csv('Customer_ID.csv')
customerData.head()


# In[5]:


transactionData = pd.read_csv('Transaction_ID.csv')
transactionData


# # Merging Datasets:

# In[6]:


# merging cabData with tansactionData to get the customer_ID column:
df1 = pd.merge(cabData, transactionData, on='Transaction ID', how='inner')


# In[7]:


df1


# In[8]:


mainData = pd.merge(df1, customerData, on='Customer ID', how='inner')
mainData.head()


# In[9]:


mainData.shape


# # 3. Data Cleaning:

# **1. Duplicated Values**
#     - we dont have any duplicated data.

# In[10]:


duplicates1 = cabData[cabData.duplicated(keep=False)]
duplicates1


# In[11]:


duplicates2 = cityData[cityData.duplicated(keep=False)]
duplicates2


# In[12]:


duplicates3 = customerData[customerData.duplicated(keep=False)]
duplicates3


# In[13]:


duplicates4 = transactionData[transactionData.duplicated(keep=False)]
duplicates4


# **1. Nulls** 
#      - we dont have any Nulls

# In[14]:


mainData.isna().sum()   # no nulls.


# **2. Formatting (Non-logic data)**

# In[15]:


# converting the Date of Travel to Actual date (Formatting) :
mainData['Date of Travel'].describe()


# In[16]:


mainData['annualDays'] = mainData['Date of Travel'] - 42371  #subtracting the min value from all date values.


# In[17]:


start_date = pd.to_datetime('2016-01-01')
mainData['actual_date'] = start_date + pd.to_timedelta(mainData['annualDays'],unit='D')


# In[18]:


mainData = mainData.drop('annualDays',axis=1)


# In[19]:


mainData


# In[20]:


mainData['actual_date'].describe() # Now we converted the date to acutal dates from 2016-01-01 to 2018-12-30


# **3. outliers**

# In[21]:


sns.boxplot(mainData['Transaction ID'])
fig, ax = plt.subplots(figsize = (6,4))
ax.scatter(mainData['Transaction ID'],mainData['Company'])
ax.set_xlabel('(Transaction Id)')
ax.set_ylabel('(company Name )')
plt.show()


# In[22]:


sns.boxplot(mainData['Date of Travel'])
fig, ax = plt.subplots(figsize = (6,4))
ax.scatter(mainData['Date of Travel'],mainData['Company'])
ax.set_xlabel('(Date of Travel)')
ax.set_ylabel('(company Name )')
plt.show()


# In[23]:


sns.boxplot(mainData['KM Travelled'])
fig, ax = plt.subplots(figsize = (6,4))
ax.scatter(mainData['KM Travelled'],mainData['Company'])
ax.set_xlabel('(KM Travelled)')
ax.set_ylabel('(company Name )')
plt.show()


# In[24]:


sns.boxplot(mainData['Price Charged'])
fig, ax = plt.subplots(figsize = (6,4))
ax.scatter(mainData['Price Charged'],mainData['Company'])
ax.set_xlabel('(Price Charged)')
ax.set_ylabel('(company Name )')
plt.show()   
# you can see the outliers here above the 1250.


# In[25]:


# Position of the Outliers
outliers = mainData[mainData['Price Charged']>= 1250]
outliers


# In[26]:


# deleting the outliers :
# 1. first :cause they are not related to any increase in Km traveled.
# 2. second :cause they are not matched by any actual increase in cost.


# In[27]:


outliers.index


# In[28]:


mainData.drop(outliers.index,inplace=True)


# In[29]:


mainData.head()


# In[30]:


mainData = mainData.reset_index(drop=True)


# In[31]:


# data visualization after deleting the outliers:
sns.boxplot(mainData['Cost of Trip'])
fig, ax = plt.subplots(figsize = (6,4))
ax.scatter(mainData['Cost of Trip'],mainData['Company'])
ax.set_xlabel('(Cost of Trip)')
ax.set_ylabel('(company Name )')
plt.show()


# # 3. Features Engineering:

# In[32]:


# creating a new feature (net_profit)
mainData['net_profit'] = mainData['Price Charged']-mainData['Cost of Trip']


# In[33]:


# creating a new feature (year) :
mainData['Year'] = mainData['actual_date'].dt.year


# In[34]:


mainData.head()


# # 4. Getting Facts:

# ### **Net Profit**

# In[35]:


mainData['Company'].unique()


# In[36]:


# getting the net profit of every company:
pinkCab = mainData[mainData['Company'] == 'Pink Cab']
pinkCabProfit = pinkCab['net_profit']
pinkCabProfit.sum()


# In[37]:


pink_2016 = pinkCab[pinkCab['Year'] == 2016]['net_profit'].sum()
pink_2017 = pinkCab[pinkCab['Year'] == 2017]['net_profit'].sum()
pink_2018 = pinkCab[pinkCab['Year'] == 2018]['net_profit'].sum()


# In[38]:


yellowCab = mainData[mainData['Company'] == 'Yellow Cab']
yellowCabProfit = yellowCab['net_profit']
yellowCabProfit.sum()


# In[39]:


yellow_2016 = yellowCab[yellowCab['Year'] == 2016]['net_profit'].sum()
yellow_2017 = yellowCab[yellowCab['Year'] == 2017]['net_profit'].sum()
yellow_2018 = yellowCab[yellowCab['Year'] == 2018]['net_profit'].sum()


# ### **Number of Customers**

# In[40]:


# getting number of customers:
totalCustomers = mainData['Customer ID']
len(totalCustomers.unique())


# In[41]:


pinkCustomers = pinkCab['Customer ID']
len(pinkCustomers.unique())


# In[42]:


yellowCustomers = yellowCab['Customer ID']
len(yellowCustomers.unique())


# ### **Number of Transactions**

# In[43]:


# getting the number of transactions for every company :
totalTrans = mainData['Transaction ID']
len(totalTrans)


# In[44]:


pinkCabTransaction = pinkCab['Transaction ID']
len(pinkCabTransaction)


# In[45]:


yearlyPinktransactions = pinkCab.groupby(['Company','Year']).size().reset_index(name='yearlyPinktransactions')
yearlyPinktransactions


# In[46]:


yellowCabTransaction = yellowCab['Transaction ID']
len(yellowCabTransaction)


# In[47]:


yearlyYellowtransactions = yellowCab.groupby(['Company','Year']).size().reset_index(name='yearlyYellowtransactions')
yearlyYellowtransactions


# ### **Total KM Travelled**

# In[48]:


# getting the number of Km travelled for every company :
totalKm = mainData['KM Travelled']
totalKm.sum()


# In[49]:


pinkCabKm = pinkCab['KM Travelled']
pinkCabKm.sum()


# In[50]:


yellowCabKm = yellowCab['KM Travelled']
yellowCabKm.sum()


# ### **Number of Transactions in each city**

# In[51]:


pinkCabCity = pinkCab['City']
len(pinkCabCity.unique())


# In[52]:


yellowCabCity = yellowCab['City']
len(yellowCabCity.unique())


# In[53]:


pinkCabTrans = pinkCab.groupby(['City']).size().reset_index(name='pinkCabTrans')
pinkCabTrans


# In[54]:


yellowCabTrans = yellowCab.groupby(['City']).size().reset_index(name='yellowCabTrans')
yellowCabTrans


# In[55]:


NewCityData = pd.merge(pinkCabTrans, yellowCabTrans, on='City', how='inner')
NewCityData ['tolalTrans']= NewCityData ['pinkCabTrans'] + NewCityData ['yellowCabTrans']
NewCityData


# ### **Number of Customers in each city**

# In[56]:


x = pinkCab.drop_duplicates(subset='Customer ID')
pinkCabCustomers = x.groupby(['City']).size().reset_index(name='pinkCabCustomers')
NewCityData = pd.merge(pinkCabCustomers, NewCityData, on='City', how='inner')
NewCityData


# In[57]:


y = yellowCab.drop_duplicates(subset='Customer ID')
yellowCabCustomers = y.groupby(['City']).size().reset_index(name='yellowCabCustomers')
NewCityData = pd.merge(yellowCabCustomers, NewCityData, on='City', how='inner')


# In[58]:


NewCityData


# In[59]:


NewCityData['totalCustomers'] = NewCityData['yellowCabCustomers'] + NewCityData['pinkCabCustomers']
NewCityData


# ### **Creating new customer dataset contains:**
# 1. Number of rides for each company per customer.
# 2. Number of mutual customers.
# 2. The company each customer used in the first and last time.

# In[60]:


customerData.head()


# In[61]:


pinkrides = pinkCab.groupby(['Customer ID']).size().reset_index(name='pinkrides')
NewCustomerData = pd.merge(pinkrides, customerData, on='Customer ID', how='inner')

q1 = pinkrides[pinkrides['pinkrides']> 5] # customers used cab for more than 5 times
r1 =  pinkrides[pinkrides['pinkrides']> 10] # customers used cab for more than 10 times
s1 =  pinkrides[pinkrides['pinkrides']> 15] # customers used cab for more than 15 times
print(len(q1))
print(len(r1))
print(len(s1))


# In[62]:


yellowrides = yellowCab.groupby(['Customer ID']).size().reset_index(name='yellowrides')
q2 = yellowrides[yellowrides['yellowrides']> 5]
r2 =  yellowrides[yellowrides['yellowrides']> 10]
s2 =  yellowrides[yellowrides['yellowrides']> 15]
print(len(q2))
print(len(r2))
print(len(s2))


# In[63]:


mutualCustomer = pd.merge(yellowrides, NewCustomerData, on='Customer ID', how='inner')
 


# In[64]:


mutualCustomer[mutualCustomer['yellowrides']<mutualCustomer['pinkrides']]


# In[65]:


mutualCustomer[mutualCustomer['yellowrides']>mutualCustomer['pinkrides']]


# In[66]:


mutualCustomer[mutualCustomer['yellowrides']==mutualCustomer['pinkrides']]


# In[67]:


mutualCustomer.info()


# In[68]:


# getting the first and last rides per customer:
largeCustomersData = mainData.sort_values(by=['Customer ID', 'actual_date'])

first_last_ride = largeCustomersData.groupby('Customer ID').agg({'Company': ['first', 'last']}).reset_index()

first_last_ride.columns = ['Customer ID', 'first_ride', 'last_ride']

first_last_ride


# In[69]:


first_last_ride[(first_last_ride['first_ride']==first_last_ride['last_ride']) & (first_last_ride['first_ride']=='Yellow Cab')]


# In[70]:


first_last_ride[(first_last_ride['first_ride']==first_last_ride['last_ride']) & (first_last_ride['first_ride']=='Pink Cab')]


# In[71]:


first_last_ride[(first_last_ride['first_ride']<first_last_ride['last_ride'])]


# # 5. Data Visualization:

# In[72]:


NetProfit = { 
    'pink' : pinkCabProfit.sum(),
    'yellow' : yellowCabProfit.sum()
}

categories = list(NetProfit.keys())

profit_counts = list(NetProfit.values())

plt.bar(categories, profit_counts)

plt.xlabel('Company')
plt.ylabel('Net Profit')
plt.title('Net Profit for each Company')

plt.show()


# In[73]:


# displaying the yearly profit for each company:

Yearly_profit = {'Pink' : [pink_2016 , pink_2017 , pink_2018],
                'Yellow': [yellow_2016 , yellow_2017 , yellow_2018]
                }
years = [2016, 2017, 2018]

plt.plot(years, Yearly_profit['Pink'], label='Pink', marker='o', color = 'r')

plt.plot(years, Yearly_profit['Yellow'], label='Yellow', marker='o', color = 'y')
plt.xticks(years)

plt.xlabel('Year')
plt.ylabel('Profit')
plt.title('Yearly Profit for Pink and Yellow')

plt.legend()

plt.grid(True)
plt.show()


# In[79]:


# displaying the total number of customers for each company:

NumberOfCustomers = {
    'Total' : len(totalTrans) , 
    'pinkCustomers' : len(pinkCabTransaction),
    'yellowCustomers' : len(yellowCabTransaction) }

customer_counts = {key: int(value) for key, value in NumberOfCustomers.items()}

categories = list(NumberOfCustomers.keys())

customer_counts = list(NumberOfCustomers.values())

plt.bar(categories, customer_counts)

plt.xlabel('Company')
plt.ylabel('Number of Customers')
plt.title('Number of Customers in each Company')

plt.show()


# In[78]:


# displaying the total number of transactions for each company:

NumberOfTrans = {
    'Total' : len(totalTrans) , 
    'pink' : len(pinkCabTransaction),
    'yellow' : len(yellowCabTransaction)
}

categories = list(NumberOfTrans.keys())

Trans_counts = list(NumberOfTrans.values())

plt.bar(categories, Trans_counts)

plt.xlabel('Company')
plt.ylabel('Number of Transactions')
plt.title('Number of Transactions in each Company')

plt.show()


# In[80]:


# displaying the yearly transactions for each company:

years = [2016, 2017, 2018]

plt.plot(years, yearlyPinktransactions['yearlyPinktransactions'], label='Pink', marker='o', color = 'r')

plt.plot(years, yearlyYellowtransactions['yearlyYellowtransactions'], label='Yellow', marker='o', color = 'y')
plt.xticks(years)

plt.xlabel('Year')
plt.ylabel('Transactions')
plt.title('Yearly Transactions for Pink and Yellow')

plt.legend()
plt.grid(True)
plt.show()


# In[81]:


# displaying the numbers of Km traveled by each company:

Km_Traveled = {
    'Pink' : pinkCabKm.sum(),
    'Yellow' : yellowCabKm.sum(),
    'Total' : totalKm.sum()
}

categories = list(Km_Traveled.keys())

Km_counts = list(Km_Traveled.values())

plt.bar(categories, Km_counts)

plt.xlabel('Company')
plt.ylabel('# Kilometers Travelled')
plt.title('# Km Travelled by each Company')
plt.show()


# In[82]:


# displaying the number of transactions for each company per city:

plt.figure(figsize=(15, 8))
# Create a stacked bar chart
plt.bar(NewCityData['City'], NewCityData['pinkCabTrans'], label='Pink Cab')
plt.bar(NewCityData['City'], NewCityData['yellowCabTrans'], label='Yellow Cab', bottom=NewCityData['pinkCabTrans'])

plt.xlabel('City')
plt.ylabel('Number of Transactions')
plt.title('Transactions by City')

plt.legend()

plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.show()


# In[83]:


# displaying the number of customers for each company per city:

plt.figure(figsize=(15, 8))

plt.bar(NewCityData['City'], NewCityData['pinkCabCustomers'], label='Pink Cab')
plt.bar(NewCityData['City'], NewCityData['yellowCabCustomers'], label='Yellow Cab', bottom=NewCityData['pinkCabCustomers'])

plt.xlabel('City')
plt.ylabel('Number of Customers')
plt.title('Customers by City')

plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.grid()
plt.show()


# In[84]:


# displaying the number of customers that used the same company more than 5 , 10 , 15 time:

numbers = {
    'Pink Cab' : [len(q1),len(r1),len(s1)],
    'Yellow Cab' : [len(q2),len(r2),len(s2)]
}

categories = ['More_than_5', 'More_than_10', 'More_than_15']
bar_width = 0.35
index = range(len(categories))

plt.bar(index, [len(q1), len(r1), len(s1)], bar_width, label='Pink Cab')
plt.bar([i + bar_width for i in index], [len(q2), len(r2), len(s2)], bar_width, label='Yellow Cab')

plt.xlabel('No of rides')
plt.ylabel('Number of Customer')
plt.title('Customers Statistics')
plt.xticks([i + bar_width / 2 for i in index], categories)

plt.legend()
plt.show()


# In[ ]:




