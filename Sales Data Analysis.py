#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly


# In[2]:


sales_data = pd.read_csv("E:\Internship Task\AfameTechnologies\sales_data.csv")


# In[3]:


sales_data.columns


# In[4]:


sales_data.shape


# In[5]:


sales_data


# In[6]:


sales_data.info()


# In[7]:


sales_data.describe()


# In[8]:


sales_data.isna().any() #checking any missing value in dataset


# In[10]:


sales_data.isna().sum() #postal column with 41296 missing values


# In[11]:


sales_data['Postal Code']


# In[12]:


sales_data['Postal Code'].unique()


# In[13]:


postal_codes = sales_data['Postal Code']

# Attempt to convert the "Postal Code" column to numeric format
try:
    postal_codes_numeric = pd.to_numeric(postal_codes)
    print("Conversion to numeric format successful.")
except ValueError as e:
    print("Error encountered during conversion to numeric format:")
    print(e)


# In[14]:


sales_data['Postal Code'] = pd.to_numeric(sales_data['Postal Code'])


# In[15]:


plt.figure(figsize=(10, 6))
sns.histplot(postal_codes, bins=30, color='skyblue')
plt.xlabel('Postal Code')
plt.ylabel('Frequency')
plt.title('Distribution of Postal Codes')
plt.show()


# In[16]:


sales_data['Postal Code']


# In[17]:


mode_postal_code = sales_data['Postal Code'].mode()[0]
sales_data['Postal Code'].fillna(mode_postal_code, inplace=True)


# In[18]:


sales_data['Postal Code'].isna().sum()


# In[19]:


sales_data['Row ID']


# In[20]:


print(sales_data['Row ID'].unique())
print(sales_data['Row ID'].nunique())


# In[21]:


sales_data['Order ID'].dtype


# In[22]:


sales_data['Order ID']


# In[23]:


print(sales_data['Order ID'].unique())
print(sales_data['Order ID'].nunique())


# In[24]:


unique_ids, counts = sales_data['Order ID'].value_counts().index, sales_data['Order ID'].value_counts().values

# Define colors for bars
colors = ['skyblue', 'salmon', 'lightgreen', 'lightcoral', 'lightblue',
          'orange', 'yellow', 'green', 'purple', 'pink']

# Plot the bar plot
plt.figure(figsize=(10, 6))
bars = plt.barh(unique_ids[:10], counts[:10], color=colors)

# Add labels and title
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Order ID', fontsize=12)
plt.title('Top 10 Most Frequent Order IDs', fontsize=14)

# Add frequency labels on each bar
for bar, count in zip(bars, counts[:10]):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{count}',
             va='center', ha='left', fontsize=10, color='black')

# Invert y-axis to display the highest frequency at the top
plt.gca().invert_yaxis()

# Show plot
plt.tight_layout()
plt.show()


# In[25]:


sales_data['Order Date']


# In[26]:


print(sales_data['Order Date'].unique())
print(sales_data['Order Date'].nunique())
     


# In[27]:


plt.figure(figsize=(10, 6))
sales_data['Order Date'].hist(bins=50, color='skyblue')
plt.xlabel('Order Date', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Orders Over Time', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better visualization
plt.show()


# In[28]:


sales_data['Ship Date']


# In[29]:


print(sales_data['Ship Date'].unique())
print(sales_data['Ship Date'].nunique())


# In[30]:


plt.figure(figsize=(10, 6))
plt.plot(sales_data['Ship Date'].value_counts().sort_index(), marker='o', color='blue', linestyle='-')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Time Series of Ship Dates', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines for better visualization
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# In[31]:


sales_data['Ship Mode']


# In[32]:


print(sales_data['Ship Mode'].unique())
print(sales_data['Ship Mode'].nunique())


# In[33]:


sns.set(style="whitegrid")

# Count the frequency of each ship mode
ship_mode_counts = sales_data['Ship Mode'].value_counts()

# Plot the bar chart using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x=ship_mode_counts.index, y=ship_mode_counts.values, palette="Blues_d")
plt.title('Frequency of Ship Modes', fontsize=14)
plt.xlabel('Ship Mode', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[34]:


sales_data['Customer ID']


# In[35]:


print(sales_data['Customer ID'].unique())
print(sales_data['Customer ID'].nunique())
     


# In[36]:


sns.set(style="whitegrid")

# Count the frequency of each customer ID
customer_id_counts = sales_data['Customer ID'].value_counts()

# Plot the bar chart using Seaborn
plt.figure(figsize=(12, 6))
barplot = sns.barplot(x=customer_id_counts.index[:10], y=customer_id_counts.values[:10], palette="husl")
plt.title('Top 10 Most Frequent Customer IDs', fontsize=14)
plt.xlabel('Customer ID', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=45)

# Adding annotations
for index, value in enumerate(customer_id_counts.values[:10]):
    barplot.text(index, value, str(value), ha="center", fontsize=10, color='black')

plt.tight_layout()
plt.show()


# In[37]:


sales_data['Customer Name']


# In[38]:


print(sales_data['Customer Name'].unique())
print(sales_data['Customer Name'].nunique()) 


# In[39]:


get_ipython().system('pip install squarify')


# In[40]:


import squarify

# Get the frequency of each customer name
customer_name_counts = sales_data['Customer Name'].value_counts()

# Selecting only the top 10 customers
top_10_customers = customer_name_counts.head(10)

# Prepare the labels with frequencies
labels = [f'{name}\n({count})' for name, count in zip(top_10_customers.index, top_10_customers.values)]

# Plotting the treemap with labeled frequencies
plt.figure(figsize=(10, 8))
squarify.plot(sizes=top_10_customers.values, label=labels, alpha=0.5)
plt.axis('off')
plt.title('Treemap of Top 10 Customer Names by Frequency')
plt.show()


# In[41]:


sales_data['Segment']


# In[42]:


print(sales_data['Segment'].unique())
print(sales_data['Segment'].nunique())


# In[43]:


sns.set(font_scale=1.2)

# Get the unique values and their counts for the Segment column
segment_counts = sales_data['Segment'].value_counts()

# Define explode values
explode = [0.1 if seg == 'Consumer' else 0 for seg in segment_counts.index]

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(x=segment_counts, labels=segment_counts.index, colors=sns.color_palette('Set2'), startangle=90, autopct='%1.2f%%', pctdistance=0.80, explode=explode)

# Add a hole in the pie
hole = plt.Circle((0, 0), 0.65, facecolor='white')
plt.gcf().gca().add_artist(hole)

plt.title('Distribution of Segments')
plt.show()


# In[44]:


sales_data['City']


# In[45]:


print(sales_data['City'].unique())
print(sales_data['City'].nunique())


# In[46]:


city_counts = sales_data['City'].value_counts()

# Plotting the count plot
plt.figure(figsize=(10, 6))
sns.countplot(y='City', data=sales_data, order=city_counts.index[:10], palette='viridis')
plt.title('Top 10 Most Frequent Cities')
plt.xlabel('Frequency')
plt.ylabel('City')
plt.show()
     


# In[47]:


sales_data['State']


# In[48]:


print(sales_data['State'].unique())
print(sales_data['State'].nunique())


# In[49]:


state_counts = sales_data['State'].value_counts()

# Plotting the count plot
plt.figure(figsize=(10, 6))
sns.countplot(y='State', data=sales_data, order=state_counts.index[:10], palette='magma')
plt.title('Top 10 Most Frequent States')
plt.xlabel('Frequency')
plt.ylabel('State')
plt.show()
     


# In[50]:


sales_data['Country']
     


# In[51]:


print(sales_data['Country'].unique())
print(sales_data['Country'].nunique())


# In[52]:


country_counts = sales_data['Country'].value_counts().head(10)

# Plotting the pie chart
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.2)
sns.color_palette("tab10")
plt.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Top 10 Countries')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[53]:


sales_data['Market']


# In[54]:


print(sales_data['Market'].unique())
print(sales_data['Market'].nunique())


# In[55]:


plt.figure(figsize=(10, 6))
sns.countplot(data=sales_data, y='Market', order=sales_data['Market'].value_counts().index, palette='viridis')
plt.title('Distribution of Markets')
plt.xlabel('Count')
plt.ylabel('Market')
plt.show()


# In[56]:


sales_data['Region'] #region column


# In[57]:


print(sales_data['Region'].unique())
print(sales_data['Region'].nunique())


# In[58]:


region_counts = sales_data['Region'].value_counts()

# Create a squarify plot
plt.figure(figsize=(10, 8))
squarify.plot(sizes=region_counts, label=region_counts.index, alpha=0.5, pad=True)
plt.title('Distribution of Regions (Treemap)')
plt.axis('off')  # Turn off axis
plt.show()
     


# In[59]:


sales_data['Product ID']


# In[60]:


print(sales_data['Product ID'].unique())
print(sales_data['Product ID'].nunique())


# In[61]:


top_n = 10

# Get the top N most frequent product IDs and their counts
top_product_ids = sales_data['Product ID'].value_counts().head(top_n)
product_counts = top_product_ids.values
product_ids = top_product_ids.index

# Set Seaborn's color palette
sns.set_palette("magma")

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.barh(product_ids, product_counts)
plt.xlabel('Count')
plt.ylabel('Product ID')
plt.title(f'Top {top_n} Most Frequent Product IDs')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest count on top
plt.show()


# In[62]:


sales_data['Category']


# In[63]:


print(sales_data['Category'].unique())
print(sales_data['Category'].nunique())


# In[67]:


from wordcloud import WordCloud

# Concatenate all categories into a single string
categories_text = ' '.join(sales_data['Category'])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(categories_text)

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Categories')
plt.axis('off')
plt.show()


# In[68]:


sales_data['Sub-Category']


# In[69]:


print(sales_data['Sub-Category'].unique())
print(sales_data['Sub-Category'].nunique())


# In[70]:


top_subcategories = sales_data['Sub-Category'].value_counts().head(10)

# Create the vertical bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top_subcategories.values, y=top_subcategories.index, palette='viridis')
plt.xlabel('Count')
plt.ylabel('Sub-Category')
plt.title('Top 10 Most Frequent Sub-Categories')
plt.show()


# In[71]:


sales_data['Product Name']


# In[72]:


print(sales_data['Product Name'].unique())
print(sales_data['Product Name'].nunique())


# In[73]:


top_product_names = sales_data['Product Name'].value_counts().head(10)

# Create the horizontal bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top_product_names.values, y=top_product_names.index, palette='magma')
plt.xlabel('Count')
plt.ylabel('Product Name')
plt.title('Top 10 Most Frequent Product Names')
plt.show()


# In[74]:


sales_data['Sales']


# In[75]:


print(sales_data['Sales'].unique())
print(sales_data['Sales'].nunique())


# In[76]:


sales_data['Quantity']


# In[77]:


print(sales_data['Quantity'].unique())
print(sales_data['Quantity'].nunique())


# In[78]:


plt.figure(figsize=(10, 6))
sns.histplot(sales_data['Quantity'], bins=14, color='skyblue')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.title('Distribution of Quantity')
plt.grid(True)
plt.show()


# In[79]:


sales_data['Discount']


# In[80]:


print(sales_data['Discount'].unique())
print(sales_data['Discount'].nunique())


# In[81]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create the KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(sales_data['Discount'], shade=True, color='skyblue')
plt.xlabel('Discount')
plt.ylabel('Density')
plt.title('Distribution of Discount')
plt.grid(True)
plt.show()
     


# In[82]:


sales_data['Profit']


# In[83]:


print(sales_data['Profit'].unique())
print(sales_data['Profit'].nunique())


# In[84]:


sales_data['Shipping Cost']


# In[85]:


print(sales_data['Shipping Cost'].unique())
print(sales_data['Shipping Cost'].nunique())


# In[86]:


sales_data['Order Priority']


# In[87]:


print(sales_data['Order Priority'].unique())
print(sales_data['Order Priority'].nunique())


# In[88]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create the count plot
plt.figure(figsize=(8, 6))
sns.countplot(x='Order Priority', data=sales_data, palette='Set3')
plt.xlabel('Order Priority')
plt.ylabel('Count')
plt.title('Distribution of Order Priority')
plt.show()


# In[89]:


sales_data.columns


# In[90]:


total_sales = sales_data['Sales'].sum()
print("Total Sales:", total_sales)


# In[91]:


total_sales = round(sales_data['Sales'].sum(), 2)
print("Total Sales:", total_sales)


# In[92]:


total_sales_by_category = sales_data.groupby('Category')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False)

# Group the data by 'Sub-Category' and calculate total sales for each sub-category
total_sales_by_subcategory = sales_data.groupby(['Category', 'Sub-Category'])['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False)

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# Plot for Total Sales by Category
sns.barplot(x='Category', y='Sales', data=total_sales_by_category, ax=axes[0], palette='Blues_d')
axes[0].set_title('Total Sales by Category')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Total Sales')
axes[0].tick_params(axis='x', rotation=45)

# Plot for Total Sales by Sub-Category
sns.barplot(x='Sales', y='Sub-Category', data=total_sales_by_subcategory, ax=axes[1], palette='Blues_d')
axes[1].set_title('Total Sales by Sub-Category')
axes[1].set_xlabel('Total Sales')
axes[1].set_ylabel('Sub-Category')

plt.tight_layout()
plt.show()


# In[93]:


sns.set_style("whitegrid")

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# Total Sales Over Time
sales_data['Order Date'] = pd.to_datetime(sales_data['Order Date'])  # Convert 'Order Date' to datetime
total_sales_over_time = sales_data.groupby(sales_data['Order Date'].dt.to_period('M'))['Sales'].sum()
total_sales_over_time.plot(ax=axes[0], marker='o', color='skyblue')
axes[0].set_title('Total Sales Over Time')
axes[0].set_xlabel('Order Date')
axes[0].set_ylabel('Total Sales')

# Total Sales by Region
total_sales_by_region = sales_data.groupby('Region')['Sales'].sum().sort_values(ascending=False)
sns.barplot(x=total_sales_by_region.index, y=total_sales_by_region.values, ax=axes[1], palette='viridis')
axes[1].set_title('Total Sales by Region')
axes[1].set_xlabel('Region')
axes[1].set_ylabel('Total Sales')
axes[1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels

plt.tight_layout()
plt.show()


# In[94]:


sns.set_style("whitegrid")

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Total Sales by Order Priority
total_sales_by_order_priority = sales_data.groupby('Order Priority')['Sales'].sum().sort_values(ascending=False)
axes[0].bar(total_sales_by_order_priority.index, total_sales_by_order_priority.values, color='skyblue')
axes[0].set_title('Total Sales by Order Priority')
axes[0].set_xlabel('Order Priority')
axes[0].set_ylabel('Total Sales')

# Total Sales by Customer Segment
total_sales_by_customer_segment = sales_data.groupby('Segment')['Sales'].sum().sort_values(ascending=False)
axes[1].bar(total_sales_by_customer_segment.index, total_sales_by_customer_segment.values, color='salmon')
axes[1].set_title('Total Sales by Customer Segment')
axes[1].set_xlabel('Customer Segment')
axes[1].set_ylabel('Total Sales')

# Total Sales by Market
total_sales_by_market = sales_data.groupby('Market')['Sales'].sum().sort_values(ascending=False)
axes[2].bar(total_sales_by_market.index, total_sales_by_market.values, color='lightgreen')
axes[2].set_title('Total Sales by Market')
axes[2].set_xlabel('Market')
axes[2].set_ylabel('Total Sales')
axes[2].tick_params(axis='x', rotation=45)  # Rotate x-axis labels

plt.tight_layout()
plt.show()


# In[95]:


total_sales_by_product = sales_data.groupby('Product Name')['Sales'].sum().sort_values(ascending=False)

# Identifying the best-selling products (top 10)
best_selling_products = total_sales_by_product.head(10)

# Displaying the best-selling products
print("Top 10 Best-Selling Products:")
print(best_selling_products)


# In[96]:


colors = sns.color_palette('pastel')[0:len(best_selling_products)]

# Plotting the donut chart
patches, texts, autotexts = plt.pie(best_selling_products, labels=best_selling_products.index, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Top 10 Best-Selling Products')

# Draw a circle in the middle to create the donut shape
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Create legend based on sales
sorted_labels = [label for _, label in sorted(zip(best_selling_products, best_selling_products.index), reverse=True)]
plt.legend(handles=patches, labels=sorted_labels, loc="center left", bbox_to_anchor=(1.1, 1.5))

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')
plt.show()


# In[105]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the DataFrame from a CSV file
# Replace 'your_file.csv' with the actual path to your CSV file
order_processing_efficiency = pd.read_csv("E:\Internship Task\AfameTechnologies\sales_data.csv")

# Assuming 'Order Priority' is a column in the CSV, set it as the index
order_processing_efficiency = order_processing_efficiency.set_index('Order Priority')

# Replace non-numeric values with zeros
order_processing_efficiency = order_processing_efficiency.apply(pd.to_numeric, errors='coerce').fillna(0)

# Convert only the numeric columns to integers
order_processing_efficiency = order_processing_efficiency.astype(int)

# Plotting
plt.figure(figsize=(10, 6))
sns.heatmap(order_processing_efficiency, annot=True, cmap='Blues', fmt='d')
plt.title('Order Processing Efficiency by Sale and Profit')
plt.xlabel('Sales')
plt.ylabel('Profit')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()


# In[ ]:




