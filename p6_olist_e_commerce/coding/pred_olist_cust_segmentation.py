# Import and set directory:
import os
os.getcwd()

# Set path:
input_path = 'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/P6_Customer_Segments_Oilst/2_Development/Input'
os.chdir(input_path)

# Import all libraries #
from datetime import datetime, date, timedelta
import sys
import numpy as np
import pandas as pd
import csv
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# Authorship:
__author__ = 'Taesun Yoo'
__email__ = 'yoots1988@gmail.com'

# Check out Python version:
print ("Python version: {}".format(sys.version))

#############################
# Part 2 - DISCOVER PHASE ###
#############################
# --- 1. Write Out List of Functions --- #
def load_file(file):
    '''load input CSVs as a dataframe'''
    return pd.read_csv(file, encoding='latin1')


def convert_dt_as_custom(df, var_name, dt_type):
    '''convert datatype on selected variables'''
    df[var_name] = df[var_name].astype(dt_type)
    return df[var_name]


def convert_dt_as_category(df):
    '''convert datatype from object to category'''
    for col in df.columns:
        if df[col].dtype.name == 'object':
            df[col] = df[col].astype('category')


def convert_dt_as_date(df, var_name, date_format):
    '''convert the column as date format'''
    df[var_name] = pd.to_datetime(df[var_name], format=date_format)


def convert_dt_as_date_object(df, col):
    '''convert datetime to object with parse date only'''
    df[col] = df[col].dt.date
    return df[col]


def join_data(df1, df2, join_type, key=None,
              left_index=None, right_index=None):
    '''merge the dataframes by a key'''
    df_join = pd.merge(df1, df2, how=join_type, on=key,
                       left_index=False, right_index=False)
    return df_join


def clean_data(raw_df):
    '''remove rows that contain duplicate columns'''
    clean_df = raw_df.drop_duplicates()
    # clean_df = raw_df.drop_duplicates(subset='customer_id')
    # clean_df = clean_df[clean_df.var_X > 0]
    # clean_df = clean_df[clean_df.var_Z <= 11000]
    return clean_df


def drop_data(df, var_list):
    '''drop variables from a dataframe'''
    df = df.drop(var_list, axis=1)
    return df


def EDA_missing_data(df):
    ''' compute missing value % on a df'''
    df_missing = pd.DataFrame(df.isnull().sum())
    df_missing.columns = ['count']
    df_missing = df_missing.sort_values(by='count', ascending=False)
    df_missing['pct'] = (df_missing['count']/len(df)) * 100
    return df_missing


def EDA_summary_stat_num(df):
    ''' compute numerical summary statistics '''
    df_stat_num = df.describe().T
    df_stat_num = df_stat_num[['count', 'min', 'mean', 'max', '25%', '50%', '75%', 'std']]
    df_stat_num = df_stat_num.sort_values(by='count', ascending=True)
    df_stat_num = pd.DataFrame(df_stat_num)
    return df_stat_num


def EDA_summary_stat_cat(df):
    ''' compute numerical summary statistics '''
    df_stat_cat = pd.DataFrame(df.describe(include='category').T)
    # df_stat_cat = pd.DataFrame(df.describe(include='O').T)
    return df_stat_cat


def feature_replacement(X):
    ''' replace missing values based on specific data type of a column '''
    for col in X.columns:
        if X[col].dtype.name == 'object':
            mode = X[col].mode().iloc[0]
            X[col] = X[col].fillna(mode)
        elif X[col].dtype.name == 'float64':
            mean = X[col].mean()
            X[col] = X[col].fillna(mean)
        elif X[col].dtype.name == 'datetime64[ns]':
            pseudo_date = pd.Timestamp.max
            X[col] = X[col].fillna(pseudo_date)
        else:
            X[col].dtype.name == 'int64'
            median = X[col].median()
            X[col] = X[col].fillna(median)


def feature_scaling(X):
    '''Feature scaled data based on standardization'''
    sc_X = StandardScaler()
    X_std = sc_X.fit_transform(X)
    return X_std


def train_model(X_cols, P1):
    ''' train the K-means cluster with specified parameters '''
    # model = KMeans(n_clusters=P1, init='k-means++', max_iter=P2, n_init=P3, random_state=0)
    kmeans = KMeans(n_clusters=P1, random_state=1)
    cluster_labels = kmeans.fit_predict(X_cols)
    return cluster_labels


def pca_transform(X_cols, N):
    '''specify N components and fit PCA'''
    pca = PCA(n_components=N)
    X_cols_pca = pca.fit_transform(X_cols)
    return X_cols_pca

def plot_pca_scatter(x,y,label,df,header):
    ''' Plot the scatter plot of PCA clusters '''
    sns.lmplot(x=x, y=y, hue=label, data=df, fit_reg=False)
    plt.title('PCA Scatter Plot: '+header, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)


def write_as_csv(df, filename, filepath):
    ''' save the dataframe as .csv file in specific filepath '''
    df.to_csv(os.path.join(filepath, filename), index=False, index_label=None)

# --- 2. Load the data --- #
# define input CSVs:
if __name__ == '__main__':
    customers = 'olist_customers.csv'
    orders = 'olist_orders.csv'
    order_items = 'olist_order_items.csv'
    order_payment = 'olist_order_payments.csv'
    products = 'olist_products.csv'
    prod_name_translator = 'olist_products_name_translator.csv'
    sellers = 'olist_sellers.csv'
    # order_reviews = 'olist_order_reviews.csv'
    # geolocation = 'olist_geolocation.csv'
    
# Load data
customers = load_file(customers)
orders = load_file(orders)
order_items = load_file(order_items)
order_payment = load_file(order_payment)
products = load_file(products)
prod_name_translator = load_file(prod_name_translator)
sellers = load_file(sellers)

# rename the columns
orders.rename(columns={'order_purchase_timestamp':'order_purchase_date',
                       'order_approved_at':'order_approved_date'}, inplace=True)    

products.rename(columns={'product_category_name':'prod_cat_name', 
                         'product_name_lenght':'prod_name_length', 
                         'product_description_lenght':'prod_desc_length',
                         'product_photos_qty':'prod_photo_qty', 
                         'product_weight_g':'prod_weight_g',
                         'product_length_cm':'prod_length_cm', 
                         'product_height_cm':'prod_height_cm',
                         'product_width_cm':'prod_width_cm'}, inplace=True)    

prod_name_translator.rename(columns={'ï»¿product_category_name':'prod_cat_name',
                                     'product_category_name_english':'prod_cat_name_eng'},inplace=True)

# --- 3. ETL data transforms ---
# left joins order_items to orders on order_id
df_join_1 = join_data(order_items, orders, 'left', 'order_id')

# left joins df_join_1 to products on product_id
df_join_2 = join_data(df_join_1, products, 'left', 'product_id')

# left joins df_join_2 to sellers on seller_id
df_join_3 = join_data(df_join_2, sellers, 'left', 'seller_id')

# left joins df_join_3 to prod_name_translator on prod_cat_name
df_join_4 = join_data(df_join_3, prod_name_translator, 'left', 'prod_cat_name')

# left joins df_join_4 to customers on customer_id
df_join_5 = join_data(df_join_4, customers, 'left', 'customer_id')

# left joins df_join_5 to order payment on order_id
olist_data = join_data(df_join_5, order_payment, 'left', 'order_id')

# delete the set of dataframes
del(customers, orders, order_items, order_payment, products, prod_name_translator, sellers)
del(df_join_1, df_join_2, df_join_3, df_join_4, df_join_5)

# check the data types on final dataframe
olist_data.dtypes

# check the column names
olist_data.columns

# --- 4. ETL data cleaning ---
# define unwanted variables
vars_unwanted = ['seller_zip_code_prefix', 'customer_zip_code_prefix', 'order_approved_date', 'order_delivered_carrier_date',
                 'order_estimated_delivery_date','shipping_limit_date',  'payment_sequential', 'payment_installments',
                 'prod_name_length', 'prod_desc_length', 'prod_photo_qty', 'prod_weight_g', 'prod_length_cm',
                 'prod_height_cm', 'prod_width_cm']

vars_dates = ['order_purchase_date', 'order_delivered_customer_date']

# drop the data - unwanted variables
olist_data = drop_data(olist_data, vars_unwanted)

# clean the data -  remove duplicates
olist_data_clean = shuffle(clean_data(olist_data)).reset_index(drop=True)

del(olist_data)

# define categorical
vars_cat = olist_data_clean.nunique()[olist_data_clean.nunique() < 28938].keys().tolist()

# from vars_cat: remove or add variables 
unwanted_items = {'price', 'freight_value'}
wanted_items = {'order_id', 'customer_id', 'customer_unique_id', 'product_id'}

# remove unwated items
vars_cat = [col for col in vars_cat if col not in unwanted_items]

# append the wanted items
vars_cat.extend(wanted_items)

# define numerical
vars_num = [col for col in olist_data_clean.columns if col not in vars_dates + vars_cat]

# format date into datetime
date_format = '%Y-%m-%d'
convert_dt_as_date(olist_data_clean, 'order_purchase_date', date_format)
convert_dt_as_date(olist_data_clean, 'order_delivered_customer_date', date_format)

# parse only dates
olist_data_clean['order_purchase_date'] = olist_data_clean['order_purchase_date'].dt.floor('d')
olist_data_clean['order_delivered_customer_date'] = olist_data_clean['order_delivered_customer_date'].dt.floor('d')

# --- 5. Feature Imputation ---
# before imputation
df_missing_pre = EDA_missing_data(olist_data_clean)
df_missing_pre

# feature imputation
feature_replacement(olist_data_clean)

# after imputation
df_missing_post = EDA_missing_data(olist_data_clean)
df_missing_post

del(df_missing_pre, df_missing_post)

# --- 6. EDA: numerical and categorical ---
olist_data_clean.info()

# convert the order_item_id as object
convert_dt_as_custom(olist_data_clean, 'order_item_id', 'object')
 
# convert data type: object to category
convert_dt_as_category(olist_data_clean)

# compute summary stat. by numerical
df_stat_num = EDA_summary_stat_num(olist_data_clean[vars_num])
df_stat_num    

# compute summary stat. by categorical
df_stat_cat = EDA_summary_stat_cat(olist_data_clean[vars_cat])
df_stat_cat 

# --- 7. RFM Analysis --- #
# check the purchase period of Olist e-commerce dataset:
print('Min:{}; Max:{}'.format(min(olist_data_clean.order_purchase_date),
                              max(olist_data_clean.order_purchase_date)))

# create the snapshot_date: max_invoice_date + 1 day
snapshot_date = max(olist_data_clean.order_purchase_date) + timedelta(days=1)

# --- calculate RFM values  ---
# Calculate Recency, Frequency and Monetary value for each customer 
rfm_datamart = olist_data_clean.groupby(['customer_unique_id']).agg({
    'order_purchase_date': lambda x: (snapshot_date - x.max()).days,
    'order_id': 'count',
    'payment_value': 'sum'})

# Rename the columns 
rfm_datamart.rename(columns={'order_purchase_date': 'Recency',
                             'order_id': 'Frequency',
                             'payment_value': 'MonetaryValue'}, inplace=True)

# Print top 5 rows
print(rfm_datamart.head())

# --- calculate 3 groups for recency, frequency and MonetaryValue ---
# Create labels for Recency and Frequency
r_labels = range(3, 0, -1); f_labels = range(1, 4); m_labels = range(1, 4)

# Assign these labels to three equal percentile groups 
r_groups = pd.qcut(rfm_datamart['Recency'], q=3, labels=r_labels)

# Assign these labels to custom pd.cut groups
f_groups = pd.cut(rfm_datamart['Frequency'], bins=3, labels=f_labels)
# f_groups = pd.qcut(rfm_datamart['Frequency'], q=3, labels=f_labels)

# Assign these labels to three equal percentile groups 
m_groups = pd.qcut(rfm_datamart['MonetaryValue'], q=3, labels=m_labels)

# Create new columns R and F 
rfm_datamart = rfm_datamart.assign(R=r_groups.values, F=f_groups.values, M=m_groups.values)

# Print top 5 rows
print(rfm_datamart.head())

del(r_labels, f_labels, m_labels)
del(r_groups, f_groups, m_groups)

# --- calculate RFM score ---
def join_rfm(x): 
    ''' Concatenate RFM quartile values to RFM_Segment'''
    return str(x['R']) + str(x['F']) + str(x['M'])

# Concatenate RFM quartile values to RFM_Segment
rfm_datamart['RFM_Segment'] = rfm_datamart.apply(join_rfm, axis=1)

# Calculate RFM_Score
rfm_datamart['RFM_Score'] = rfm_datamart[['R','F','M']].sum(axis=1)
print(rfm_datamart['RFM_Score'].head())

# --- Find average monetary value for RFM score segment equals 9 ---
print(rfm_datamart[rfm_datamart.RFM_Score == 9].MonetaryValue.mean())

# --- analyzing customer groups per RFM_Score ---
rfm_score_agg = rfm_datamart.groupby('RFM_Score').agg({
    'Recency': 'mean',
    'Frequency': 'mean',  
    'MonetaryValue': ['mean', 'count']
}).round(1)

print(rfm_score_agg)

# --- creating custom segment labels ---
# Define rfm_level function
def rfm_level(df):
    ''' assign a custome segment label based on RFM score '''
    if df['RFM_Score'] >= 7:
        return '1_High'
    elif ((df['RFM_Score'] >= 5) and (df['RFM_Score'] < 7)):
        return '2_Medium'
    else:
        return '3_Low'

# Create a new variable RFM_Level
rfm_datamart['RFM_Level'] = rfm_datamart.apply(rfm_level, axis=1)

# Print top 5 rows
print(rfm_datamart.head())

# --- analyzing custom segments ---
# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_level_agg = rfm_datamart.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',  
    'MonetaryValue': ['mean', 'count']
}).round(1)

# Print the aggregated dataset
print(rfm_level_agg)

# --- 8. Data Pre-Processing --- #
# --- calculate statistics of variables  ---
# Print the average values of the variables in the dataset
print(rfm_datamart.mean())

# Print the standard deviation of the variables in the dataset
print(rfm_datamart.std())

# Get the key statistics of the dataset
print(rfm_datamart.describe())

# confirmed that no equal mean and variance on R-F-M segment(s)

# --- detect skewed variables ---
# Plot distribution of Recency
plt.subplot(3, 1, 1); sns.distplot(rfm_datamart['Recency'])

# Plot distribution of Frequency
plt.subplot(3, 1, 2); sns.distplot(rfm_datamart['Frequency'])

# Plot distribution of MonetaryValue
plt.subplot(3, 1, 3); sns.distplot(rfm_datamart['MonetaryValue'])

# Show the plot
plt.show()

# # --- managed skewness  ---
# # Apply log transformation to Recency
# def feature_log(df, col):
#     ''' apply logarithmic function to a specified column '''
#     df[col + str('_log')] = np.log(df[col])
#     return df[col + str('_log')]

# rfm_datamart['Recency_log'] = feature_log(rfm_datamart, 'Recency')
# rfm_datamart['Frequency_log'] = feature_log(rfm_datamart, 'Frequency')
# rfm_datamart['MonetaryValue_log'] = feature_log(rfm_datamart, 'MonetaryValue')

# # Create a subplot of the distribution of Recency_log
# plt.subplot(3, 1, 1); sns.distplot(rfm_datamart['Recency_log'])

# # Create a subplot of the distribution of Frequency_log
# plt.subplot(3, 1, 2); sns.distplot(rfm_datamart['Frequency_log'])

# # Create a subplot of the distribution of MonetaryValue_log
# plt.subplot(3, 1, 3); sns.distplot(rfm_datamart['MonetaryValue_log'])

# # Show the plot
# plt.show()

# --- pre-process RFM data ---
# Unskew the datamart with log transform 
rfm_datamart_log = np.log(rfm_datamart.iloc[:, 0:3])

# Apply feature scaler and fit_transform
rfm_datamart_normalized = feature_scaling(rfm_datamart_log)

# Create a pandas DataFrame
rfm_datamart_normalized = pd.DataFrame(rfm_datamart_normalized, index=rfm_datamart_log.index, columns=rfm_datamart_log.columns)

del(rfm_datamart_log)

# Print summary statistics
print(rfm_datamart_normalized.describe().round(2))

# Create a subplot of the distribution of Recency
plt.subplot(3, 1, 1); sns.distplot(rfm_datamart_normalized['Recency'])

# Create a subplot of the distribution of Frequency
plt.subplot(3, 1, 2); sns.distplot(rfm_datamart_normalized['Frequency'])

# Create a subplot of the distribution of MonetaryValue
plt.subplot(3, 1, 3); sns.distplot(rfm_datamart_normalized['MonetaryValue'])

# Show the plot
plt.show()

# --- 9. K-Means Clustering--- #
# --- calculate sum of squared errors ---
# Fit KMeans and calculate SSE for each k
sse = {}
for k in range(1, 11):
  
    # Initialize KMeans with k clusters
    kmeans = KMeans(n_clusters=k, random_state=1)
    
    # Fit fit on the normalized dataset
    kmeans.fit(rfm_datamart_normalized)
    
    # Assign sum of squared distances to k element of dictionary
    sse[k] = kmeans.inertia_

# --- plot sum of sqaured errors: elbow method ---
# Add the plot title "The Elbow Method"
plt.figure(figsize=(10,10))
plt.title('The Elbow Method')

# Add X-axis label "k"
plt.xlabel('k')

# Add Y-axis label "SSE"
plt.ylabel('SSE')

# Plot SSE values for each key in the dictionary
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()

# Fit KMeans and extract cluster labels
cluster_labels = train_model(rfm_datamart_normalized, 3)

# Create a DataFrame by adding a new cluster label column
rfm_datamart_k3 = rfm_datamart.assign(Cluster=cluster_labels)
del(rfm_datamart)

# Group the data by cluster
grouped_clust_3 = rfm_datamart_k3.groupby(['Cluster'])

# Calculate average RFM values and segment sizes per cluster value
grouped_clust_3.agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']
  }).round(1)

# --- 10. visualization: snake plot with 3 clusters --- #
# Compute a cluster on normalized RFM dataframe
rfm_datamart_normalized['Cluster'] = rfm_datamart_k3['Cluster']

# prepare data for the snake plot
rfm_datamart_melt = rfm_datamart_normalized[['Recency', 'Frequency', 'MonetaryValue','Cluster']]

# Melt the normalized dataset and reset the index
rfm_datamart_melt = pd.melt(rfm_datamart_melt.reset_index(),                        
                    id_vars=['customer_unique_id', 'Cluster'],
                    value_vars=['Recency', 'Frequency', 'MonetaryValue'],                         
                    var_name='Metric', 
                    value_name='Value')

# Plot a snake plot with 3 clusters
plt.figure(figsize=(10,10))
plt.title('Snake plot of standardized variables with 3 clusters')
plt.xlabel('Metric')
plt.ylabel('Value')
sns.lineplot(data=rfm_datamart_melt, x='Metric', y='Value', hue='Cluster')
plt.show()

# --- 11. visualization: relative importance plot with 3 clusters --- #
# Calculate average RFM values for each cluster
avg_cluster = rfm_datamart_k3[['Recency','Frequency','MonetaryValue','Cluster']].groupby(['Cluster']).mean() 

# Calculate average RFM values for the total customer population
avg_population = rfm_datamart_k3[['Recency','Frequency','MonetaryValue']].mean()

# Calculate relative importance of cluster's attribute value compared to population
relative_imp = (avg_cluster / avg_population) - 1

# Print relative importance scores rounded to 2 decimals
print(relative_imp.round(2))

# Initialize a plot with a figure size of 8 by 4 inches 
plt.figure(figsize=(10, 10))

# Add the plot title
plt.title('Relative importance of attributes by 3 clusters')

# Plot the heatmap
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
plt.show()

# --- 12. apply PCA decomposition: 3 clusters ---
# Subset features and label for PCA decomposition with 3 clusters
#features_k3_pca = rfm_datamart_k3[['Recency','Frequency','MonetaryValue']]
features_k3_pca = rfm_datamart_normalized[['Recency','Frequency','MonetaryValue']]

# fit and transform into PCA components with n = 2
features_k3_pca = pca_transform(features_k3_pca, 2)

# predict the cluster labels with PCA transformed features
labels_k3_pca = train_model(features_k3_pca, 3)

# create the PCA decomposed dataframe
df_k3_pca = pd.DataFrame(features_k3_pca, columns=['PCA_X','PCA_Y'])

# bind the predicted label:
df_k3_pca = df_k3_pca.assign(Cluster_PCA = labels_k3_pca)

# bind the customer_unique_id on df_pca
df_k3_pca = df_k3_pca.assign(id = rfm_datamart_k3.index)

# plot a scatter plot: visualize the PCA transformed K-mean cluster
plot_pca_scatter('PCA_X','PCA_Y','Cluster_PCA', df_k3_pca, 'K-Means [3 clusters]')


# --- 13. build modeling with 4 segments ---
olist_data_clean['order_purchase_date_2'] = olist_data_clean['order_purchase_date']

# Calculate Recency, Frequency, Monetary value and Tenure for each customer 
rfmt_datamart = olist_data_clean.groupby(['customer_unique_id']).agg({
    'order_purchase_date': lambda x: (snapshot_date - x.max()).days,
    'order_id': 'count',
    'payment_value': 'sum',
    'order_purchase_date_2': lambda x: (x.max() - x.min()).days})

# Rename the columns 
rfmt_datamart.rename(columns={'order_purchase_date': 'Recency',
                              'order_id': 'Frequency',
                              'payment_value': 'MonetaryValue',
                              'order_purchase_date_2': 'Tenure'}, inplace=True)

# dealt with rows when Tenure == 0: replaced by 1
rfmt_datamart['Tenure'] = np.where(rfmt_datamart['Tenure'] == 0, 1, rfmt_datamart['Tenure'])

# apply log transform
rfmt_datamart_log = np.log(rfmt_datamart)

# Apply feature scaler and fit_transform
rmft_datamart_normalized = feature_scaling(rfmt_datamart_log)

# Create a pandas DataFrame
rfmt_datamart_normalized = pd.DataFrame(rmft_datamart_normalized, 
                                        index=rfmt_datamart_log.index,
                                        columns=rfmt_datamart_log.columns)

# Print top 5 rows
print(rfmt_datamart_normalized.head())

# Re-fit KMeans and extract cluster labels
cluster_labels = train_model(rfmt_datamart_normalized, 4)

# Create a new DataFrame by adding a cluster label column to datamart_rfmt
rfm_datamart_k4 = rfmt_datamart.assign(Cluster=cluster_labels)

del(rfmt_datamart)

# Group by cluster
grouped_clust_4 = rfm_datamart_k4.groupby(['Cluster'])

# Calculate average RFMT values and segment sizes for each cluster
grouped_clust_4.agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': 'mean',
    'Tenure': ['mean', 'count']
  }).round(1)

# --- 14. visualization: snake plot with 4 clusters --- #
# Compute a cluster on normalized RFMT dataframe
rfmt_datamart_normalized['Cluster'] = rfm_datamart_k4['Cluster']

# prepare data for the snake plot
rfmt_datamart_melt = rfmt_datamart_normalized[['Recency', 'Frequency', 'MonetaryValue', 'Tenure', 'Cluster']]

# Melt the normalized dataset and reset the index
rfmt_datamart_melt = pd.melt(rfmt_datamart_melt.reset_index(),                        
                    id_vars=['customer_unique_id', 'Cluster'],
                    value_vars=['Recency', 'Frequency', 'MonetaryValue', 'Tenure'],                         
                    var_name='Metric', 
                    value_name='Value')

# Plot a line for each value of the cluster variable
plt.figure(figsize=(10,10))
plt.title('Snake plot of standardized variables with 4 clusters')
plt.xlabel('Metric')
plt.ylabel('Value')
sns.lineplot(data=rfmt_datamart_melt, x='Metric', y='Value', hue='Cluster')
plt.show()

# --- 15. visualization: relative importance plot with 4 clusters --- #
# Calculate average RFM values for each cluster
avg_cluster_k4 = rfm_datamart_k4[['Recency','Frequency','MonetaryValue','Cluster']].groupby(['Cluster']).mean() 

# Calculate average RFM values for the total customer population
avg_population_k4 = rfm_datamart_k4[['Recency','Frequency','MonetaryValue']].mean()

# Calculate relative importance of cluster's attribute value compared to population
relative_imp_k4 = (avg_cluster_k4 / avg_population_k4) - 1

# Print relative importance scores rounded to 2 decimals
print(relative_imp_k4.round(2))

# Initialize a plot with a figure size of 10 by 10 inches 
plt.figure(figsize=(10,10))

# Add the plot title
plt.title('Relative importance of attributes by 4 clusters')

# Plot the heatmap
sns.heatmap(data=relative_imp_k4, annot=True, fmt='.2f', cmap='RdYlGn')
plt.show()

# --- 16. apply PCA decomposition: 4 clusters ---
# Subset features and label for PCA decomposition with 4 clusters
#features_k4_pca = rfm_datamart_k4[['Recency','Frequency','MonetaryValue']]
features_k4_pca = rfmt_datamart_normalized[['Recency','Frequency','MonetaryValue']]

# fit and transform into PCA components with n = 2
features_k4_pca = pca_transform(features_k4_pca, 2)

# predict the cluster labels with PCA transformed features
labels_k4_pca = train_model(features_k4_pca, 4)

# create the PCA decomposed dataframe
df_k4_pca = pd.DataFrame(features_k4_pca, columns=['PCA_X','PCA_Y'])

# bind the predicted label:
df_k4_pca = df_k4_pca.assign(Cluster_PCA = labels_k4_pca)

# bind the customer_unique_id on df_pca
df_k4_pca = df_k4_pca.assign(id = rfm_datamart_k4.index)

# plot a scatter plot: visualize the PCA transformed K-mean cluster
plot_pca_scatter('PCA_X','PCA_Y','Cluster_PCA', df_k4_pca, 'K-Means [4 clusters]')

# --- 17. save prediction results for PCA with 3 clusters ---
# create an id column in rfm_datamart_k3, rfm_datamart_k4
rfm_datamart_k3 = rfm_datamart_k3.assign(id=rfm_datamart_k3.index)
rfm_datamart_k4 = rfm_datamart_k4.assign(id=rfm_datamart_k4.index)

# join dataframes for 3 and 4 solutions
df_review_k3 = join_data(df_k3_pca, rfm_datamart_k3, 'inner', 'id')
df_review_k4 = join_data(df_k4_pca, rfm_datamart_k4, 'inner', 'id')

vars_required = ['id', 'Recency', 'Frequency', 'MonetaryValue',
                 'Cluster', 'PCA_X', 'PCA_Y', 'Cluster_PCA']

# subset only required columns
df_review_k3 = df_review_k3[vars_required]
df_review_k4 = df_review_k4[vars_required]

# specify the output_path
output_path = r'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/P6_Customer_Segments_Oilst/2_Development/Output'

# save predictions for 3 and 4 segments
write_as_csv(df_review_k3, 'pred_cust_pca_km_3.csv', output_path)
write_as_csv(df_review_k4, 'pred_cust_pca_km_4.csv', output_path)