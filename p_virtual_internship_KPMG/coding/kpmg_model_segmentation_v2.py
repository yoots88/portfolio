# Import and set directory:
import os
os.getcwd()

# Set input path
input_path = 'D:/Virtual_Internships/1_KPMG_Data Analytics/Input'
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

# Authorization #
__author__ = "Taesun Yoo"
__email__ = "yoots1988@gmail.com"

#################################
# Part 2 - Discover the Problem #
#################################
# --- 1. Write Out List of Functions --- #
def load_file_excel(file, sheet_name):
    '''load input XLSs as a dataframe'''
    return pd.read_excel(file, sheet_name=sheet_name)


def drop_data(df, var_list):
    ''' drop variables based on the specified list '''
    df = df.drop(var_list, axis=1)
    return df


def drop_duplicates_data(data, col_id=None):
    '''subset data with non-duplicates'''
    # sort by a specified id
    data.sort_values(col_id, inplace=True)
    # make a bool series for duplicates
    bool_series_dups = data.duplicated(keep=False)
    # subset a dataframe with non-duplicate
    data_clean = data[~bool_series_dups]
    # return a subset dataframe
    return data_clean


def convert_dt_as_custom(df,var_name,var_type):
    ''' convert the data type of a speicfic variable '''
    df[var_name] = df[var_name].astype(var_type)
    return df[var_name]


def convert_dt_as_cat(df):
    ''' convert data type object to category '''
    for col in df.columns:
        if df[col].dtype.name == 'object':
            df[col] = df[col].astype('category')
            

def convert_int_to_datetime(df, col):
    '''convert integer to datetime'''
    df[col] = pd.to_datetime(df[col], unit='d', origin='1900-01-01', errors='coerce')
    return df[col]


def join_data(df1, df2, join_type, key=None,
              left_index=None, right_index=None):
    '''merge the dataframes by a key'''
    df_join = pd.merge(df1, df2, how=join_type, on=key,
                       left_index=False, right_index=False)
    return df_join


def eda_missing(df):
    '''compute missing % on each var'''
    df_missing = pd.DataFrame(df.isnull().sum(), columns=['count'])
    df_missing['pct'] = (df_missing['count']/len(df)) * 100
    return df_missing


def eda_stat_cat(df):
    ''' perform eda for categorical features '''
    df_stat_cat = df.describe(include='category').T
    df_stat_cat = pd.DataFrame(df_stat_cat)
    return df_stat_cat


def eda_stat_num(df):
    ''' perform eda for numerical features '''
    df_stat_num = df.describe().T
    df_stat_num = df_stat_num[['count', 'min', 'mean', 'max', 'std', '25%', '50%', '75%']]
    df_stat_num = pd.DataFrame(df_stat_num)
    return df_stat_num


def eda_outliers(df):
    '''check outliers using the IQR method'''
    df['IQR'] = df['75%'] - df['25%']
    df['LB']  = df['25%'] - 1.5*df['IQR']
    df['UB']  = df['75%'] + 1.5*df['IQR']
    df = df.drop(['count','std','mean','25%','50%','75%','IQR'], axis=1)
    return df


def list_comparator(l2, l1):
    ''' compare the two lists and return the difference '''
    diff_list = list(set(l2) - set(l1))
    return diff_list


def gender_conditions(df):
    '''return a list with series'''
    conditions = [df['gender'] == 'F', 
              df['gender'] == 'Female',
              df['gender'] == 'Femal',
              df['gender'] == 'U',
              df['gender'] == 'M',
              df['gender'] == 'Male']
    return conditions


def deceased_conditions(df):
    '''return a list with series'''
    conditions = [df['deceased_indicator'] == 'Y', 
              df['deceased_indicator'] == 'N']
    return conditions


def order_conditions(df):
    '''return a list with series'''
    conditions = [df['online_order'] == 1, 
              df['online_order'] == 0]
    return conditions


def state_conditions(df):
    '''return a list with series'''
    conditions = [df['state'] == 'New South Wales', 
              df['state'] == 'NSW',
              df['state'] == 'Victoria', 
              df['state'] == 'QLD',
              df['state'] == 'VIC']
    return conditions


def convert_DOB_to_age(df, var_name):
    ''' calculate age from DOB '''
    df[var_name] = np.where(df[var_name] > datetime.now(),
                            df[var_name] - timedelta(365*100),
                            df[var_name])
    df['age'] = datetime.now().year - df[var_name].apply(lambda x:x.year)


def feature_replacement(X):
    ''' replace missing values based on specific data type of a column '''
    for col in X.columns:
        if X[col].dtype.name == 'category':
            mode = X[col].mode().iloc[0]
            X[col] = X[col].fillna(mode)
        elif X[col].dtype.name == 'datetime64[ns]':
            pseudo_date = pd.Timestamp.max
            X[col] = X[col].fillna(pseudo_date)
        elif X[col].dtype.name == 'float64':
            mean = X[col].mean()
            X[col] = X[col].fillna(mean)
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
    df.to_csv(os.path.join(filepath, filename), index=False)
    

# --- 2. Load the data --- #
# define input CSVs:
if __name__ == '__main__':
    file_name = 'kpmg_dataset.xlsx'

# load dataframe by each sheet
df_trans = load_file_excel(file_name, 'Transactions')
df_new_cust = load_file_excel(file_name, 'NewCustomerList')
df_cust_demo = load_file_excel(file_name, 'CustomerDemographic')
df_cust_add = load_file_excel(file_name, 'CustomerAddress')

# check data types on dataframes
df_trans.dtypes
df_new_cust.dtypes
df_cust_demo.dtypes
df_cust_add.dtypes

# --- check duplicates ---
df_cust_add_clean = drop_duplicates_data(df_cust_add,'customer_id')
df_cust_demo_clean = drop_duplicates_data(df_cust_demo,'customer_id')
df_new_cust_clean = drop_duplicates_data(df_new_cust,'first_name')
df_trans_clean = drop_duplicates_data(df_trans,'customer_id')

del(df_cust_add, df_cust_demo, df_new_cust, df_trans)

# --- check missing values ---
df_missing_cust_add = eda_missing(df_cust_add_clean)
df_missing_cust_demo = eda_missing(df_cust_demo_clean)
df_missing_new_cust = eda_missing(df_new_cust_clean)
df_missing_trans = eda_missing(df_trans_clean)

# print missing % dataframes
print(df_missing_cust_add)

print(df_missing_cust_demo)

print(df_missing_new_cust)

print(df_missing_trans)

del(df_missing_cust_add, df_missing_cust_demo, df_missing_new_cust, df_missing_trans)

# --- 3. ETL: Join Data --- #
# prepare a train set: join customer_demographic with customer address
df_train = join_data(df_cust_demo_clean, df_cust_add_clean, 'left', 'customer_id')

# joins df_train with df_trans_clean
df_train = join_data(df_trans_clean, df_train, 'left', 'customer_id')

# prepare a test set: join new_customer_list with customer address
df_test = join_data(df_new_cust_clean, df_cust_add_clean, 'left', ['postcode','address'])

# joins df_test with df_trans_clean
df_test = join_data(df_test, df_trans_clean, 'left', 'customer_id')

# Subset dataframe based on list of columns
list_trans = list(df_trans_clean.columns)
list_address = list(df_cust_add_clean.columns)
list_new_cust = list(df_new_cust_clean.columns)
list_cust_demo = list(df_cust_demo_clean.columns)

# Specify unnamed columns
list_drop = ['Unnamed: 16', 'Unnamed: 19', 'Unnamed: 18', 'Unnamed: 17', 'Unnamed: 20']

# Clean the list for new customers
list_new_cust = [item for item in list_new_cust if item not in list_drop]

del(df_trans_clean, df_cust_add_clean, df_new_cust_clean, df_cust_demo_clean)

# --- 4. ETL: Data Cleaning --- #
############
# test set #
############
# remove duplicated columns: 'state_y','country_y'
df_test = drop_data(df_test, ['state_y','country_y','property_valuation_y'])

# drop columns: unnamed
df_test = drop_data(df_test, ['Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18',
                        'Unnamed: 19', 'Unnamed: 20'])

# rename columns: 'state_x', 'country_x'
df_test.rename(columns={'state_x':'state','country_x':'country',
                        'property_valuation_x':'property_valuation'}, inplace=True)

# add the empty column(s): default
df_test = df_test.assign(default='')

# create a column list
col_list_1 = df_test.columns.tolist()

#############
# train set #
#############
# add the empty column(s): Rank, Value
df_train = df_train.assign(Rank='')
df_train = df_train.assign(Value='')

# create a column list
col_list_2 = df_train.columns.tolist()

# compare the lists
list_comparator(col_list_2, col_list_1)

# re-arrange the columns using col_list
df_train = df_train[col_list_1]
del(col_list_1, col_list_2)

# --- ETL: Feature Encoding ---
# gender: hard-coding
choices = ['Female', 'Female', 'Female', 'Unknown', 'Male', 'Male']
conditions_train = gender_conditions(df_train)

# apply on both train and test sets
df_train['gender'] = np.select(conditions_train, choices, default=None)

# deceased_indicator: hard-coding
choices = ['Yes', 'No']

conditions_train = deceased_conditions(df_train)

# apply on both train and test sets
df_train['deceased_indicator'] = np.select(conditions_train, choices, default=None)

# online_order: hard-coding
choices = ['Yes', 'No']

conditions_train = order_conditions(df_train)

# apply on both train and test sets
df_train['online_order'] = np.select(conditions_train, choices, default=None)

# state: hard-coding
choices = ['New South Wales', 'New South Wales', 'Victoria', 'Queensland', 'Victoria']

conditions_train = state_conditions(df_train)

# apply on both train and test sets
df_train['state'] = np.select(conditions_train, choices, default=None)

del(conditions_train)

# --- ETL: Feature Engineering ---
# calculate age from DOB #
convert_DOB_to_age(df_train, 'DOB')
    
# calculate age group from age
df_train['age'].describe()

# remove any observations where age is greater than 90
df_train = df_train[df_train.age < 90]

# set up age group bins and labels
bins_age = [10, 19, 29, 39, 49, 59, 69, 79, 89]
labels_age = ['10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s']

# calculate age group by decades
df_train['age_group'] = pd.cut(x=df_train['age'], bins=bins_age, labels=labels_age)

del(bins_age, labels_age)

# --- ETL: Data Conversion ---
# convert data type: object to category
convert_dt_as_cat(df_train)
convert_int_to_datetime(df_train, 'product_first_sold_date')

# --- ETL: Feature Imputation ---
# check missing % 
eda_missing(df_train)

# replace missing values 
feature_replacement(df_train)
del(list_address, list_cust_demo, list_new_cust, list_drop, list_trans)

# make a copy of df_train as sales_data
sales_data = df_train.copy()
del(df_train, df_test)

# --- RFM Analysis ---
# check the purchase period of dataset:
print('Min:{}; Max:{}'.format(min(sales_data.transaction_date),
                              max(sales_data.transaction_date)))

# create the snapshot_date: max_invoice_date + 1 day
snapshot_date = max(sales_data.transaction_date) + timedelta(days=1)

# --- calculate RFM values  ---
# Calculate Recency, Frequency and Monetary value for each customer 
rfm_datamart = sales_data.groupby(['customer_id']).agg({
    'transaction_date': lambda x: (snapshot_date - x.max()).days,
    'transaction_id': 'count',
    'list_price': 'sum'})

# Rename the columns 
rfm_datamart.rename(columns={'transaction_date': 'Recency',
                             'transaction_id': 'Frequency',
                             'list_price': 'MonetaryValue'}, inplace=True)

# Print top 5 rows
print(rfm_datamart.head())

# --- calculate 4 groups for recency, frequency and MonetaryValue ---
# Create labels for Recency and Frequency
r_labels = range(4, 0, -1); f_labels = range(1, 5); m_labels = range(1, 5)

# Assign these labels to three equal percentile groups 
r_groups = pd.qcut(rfm_datamart['Recency'], q=4, labels=r_labels)

# Assign these labels to custom pd.cut groups
f_groups = pd.qcut(rfm_datamart['Frequency'], q=4, labels=f_labels)

# Assign these labels to three equal percentile groups 
m_groups = pd.qcut(rfm_datamart['MonetaryValue'], q=4, labels=m_labels)

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

# --- Find average monetary value for RFM score segment equals 12 ---
print(rfm_datamart[rfm_datamart.RFM_Score == 12].MonetaryValue.mean())

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
    if df['RFM_Score'] >= 9:
        return '1_High'
    elif ((df['RFM_Score'] >= 5) and (df['RFM_Score'] < 9)):
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

# --- detect skewed variables ---
# confirmed that no equal mean and variance on R-F-M segment(s)
plt.figure(figsize=(10,10))

# Plot distribution of Recency
plt.subplot(3, 1, 1); sns.distplot(rfm_datamart['Recency'])

# Plot distribution of Frequency
plt.subplot(3, 1, 2); sns.distplot(rfm_datamart['Frequency'])

# Plot distribution of MonetaryValue
plt.subplot(3, 1, 3); sns.distplot(rfm_datamart['MonetaryValue'])

# Show the plot
plt.show()

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

plt.figure(figsize=(10,10))

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
                    id_vars=['customer_id', 'Cluster'],
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
features_k3_pca = rfm_datamart_normalized[['Recency','Frequency','MonetaryValue']]

# fit and transform into PCA components with n = 2
features_k3_pca = pca_transform(features_k3_pca, 2)

# predict the cluster labels with PCA transformed features
labels_k3_pca = train_model(features_k3_pca, 3)

# create the PCA decomposed dataframe
df_k3_pca = pd.DataFrame(features_k3_pca, columns=['PCA_X','PCA_Y'])

# bind the predicted label:
df_k3_pca = df_k3_pca.assign(Cluster_PCA = labels_k3_pca)

# bind the customer_id on df_pca
df_k3_pca = df_k3_pca.assign(id = rfm_datamart_k3.index)

# plot a scatter plot: visualize the PCA transformed K-mean cluster
plot_pca_scatter('PCA_X','PCA_Y','Cluster_PCA', df_k3_pca, 'K-Means [3 clusters]')

# --- 13. build modeling with 4 segments ---
sales_data['transaction_date_2'] = sales_data['transaction_date']

# Calculate Recency, Frequency, Monetary value and Tenure for each customer 
rfmt_datamart = sales_data.groupby(['customer_id']).agg({
    'transaction_date': lambda x: (snapshot_date - x.max()).days,
    'transaction_id': 'count',
    'list_price': 'sum',
    'transaction_date_2': lambda x: (x.max() - x.min()).days})

# Rename the columns 
rfmt_datamart.rename(columns={'transaction_date': 'Recency',
                              'transaction_id': 'Frequency',
                              'list_price': 'MonetaryValue',
                              'transaction_date_2': 'Tenure'}, inplace=True)

# dealt with rows when Tenure == 0: replaced by 0.01
rfmt_datamart['Tenure'] = np.where(rfmt_datamart['Tenure'] == 0, 0.01, rfmt_datamart['Tenure'])

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
                    id_vars=['customer_id', 'Cluster'],
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

# Plot the heatmap
plt.figure(figsize=(10,10))
plt.title('Relative importance of attributes by 4 clusters')
sns.heatmap(data=relative_imp_k4, annot=True, fmt='.2f', cmap='RdYlGn')
plt.show()

# --- 16. apply PCA decomposition: 4 clusters ---
# Subset features and label for PCA decomposition with 4 clusters
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

# --- 17. save prediction results for PCA with 3 and 4 clusters ---
# create an id column in rfm_datamart_k3, rfm_datamart_k4
rfm_datamart_k3 = rfm_datamart_k3.assign(id=rfm_datamart_k3.index)
rfm_datamart_k4 = rfm_datamart_k4.assign(id=rfm_datamart_k4.index)

# join dataframes for 3 and 4 solutions
df_review_k3 = join_data(df_k3_pca, rfm_datamart_k3, 'inner', 'id')
df_review_k4 = join_data(df_k4_pca, rfm_datamart_k4, 'inner', 'id')

#vars_required = ['id', 'Recency', 'Frequency', 'MonetaryValue',
#                 'Cluster', 'PCA_X', 'PCA_Y', 'Cluster_PCA']

# subset only required columns
# df_review_k3 = df_review_k3[vars_required]
# # df_review_k4 = df_review_k4[vars_required]

# specify the filepath
output_path = r'D:/Virtual_Internships/1_KPMG_Data Analytics/Output'

# save the prediction results
write_as_csv(df_review_k3, 'kpmg_customers_rfm_segments_3_v2.csv', output_path)
# write_as_csv(df_review_k4, 'kpmg_customers_rfm_segments_4.csv', output_path)