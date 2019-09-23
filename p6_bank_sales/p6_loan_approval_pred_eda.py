# Import and set directory:
import os
os.getcwd()

# Set path:
#path = 'D:/P_Poverty_Prediction/A. Coding'
path = 'D:/TY Workspace/3. Data Science Portfolio/p6_bank_sales_prediction/A. Coding'
os.chdir(path)

# Import all libraries #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# Authorization #
__author__ = "Taesun Yoo"
__email__ = "yoots1988@gmail.com"

# --- 2. Write Out List of Functions --- #
#################################
# Part 2 - Discover the Problem #
#################################
# Write a group of funtions:
def load_file(file):
    '''load input CSVs as a dataframe'''
    df = pd.read_csv(file)
    return df

def clean_data(raw_df):
    '''clean and drop duplicates'''
    clean_df = raw_df.drop_duplicates(subset='ID')
    return clean_df

def EDA_missing_data(df):
    '''compute missing % on features of a dataframe'''
    missing_df = pd.DataFrame(df.isnull().sum())
    missing_df.columns = ['count']
    missing_df['pct'] = missing_df['count']/len(df)
    return missing_df

def EDA_numerical_data(df):
    '''compute summary stat on numerical feature(s)'''
    df_stat_num = df.describe().T
    df_stat_num = pd.DataFrame(df_stat_num[['count', 'std', 'min', 'mean', 'max', '25%', '50%', '75%']])
    return df_stat_num

def EDA_categorical_data(df):
    '''compute summary stat on categorical feature(s)'''
    df_stat_cat = pd.DataFrame(df.describe(include='O').T)
    return df_stat_cat 

def EDA_plot_num_var(df, num_var):
    '''plot a boxplot and distribution plot for numerical variable'''
    plt.figure(figsize=(14,7))
    plt.subplot(1,2,1)
    sns.boxplot(df[num_var])
    plt.subplot(1,2,2)
    sns.distplot(df[num_var], bins=20)
    plt.show()

def EDA_plot_mean_label_barchart(df, num_var1, num_var2, num_var3,
                                 label, lab_list, lab_name):
    '''plot 1 by 3 average numerical bar charts categorized by stroke labels'''
    plt.figure(figsize=(10,7))
    plt.subplots_adjust(hspace=1.3, wspace=1.3)
    # 1st plot factored by a stroke label:
    plt.subplot(1,3,1)
    df.groupby(label)[num_var1].mean().sort_values(ascending=False).plot(kind='bar')
    plt.xticks(lab_list, lab_name, rotation=0, fontsize=12)
    plt.xlabel('approval status')
    plt.ylabel('Mean ' + num_var1)
    plt.title('Mean ' + num_var1 + ' by approval status', fontsize=12)
    # 1st plot factored by a stroke label:
    plt.subplot(1,3,2)
    df.groupby(label)[num_var2].mean().sort_values(ascending=False).plot(kind='bar')
    plt.xticks(lab_list, lab_name, rotation=0, fontsize=12)
    plt.xlabel('approval status')
    plt.ylabel('Mean ' + num_var2)
    plt.title('Mean ' + num_var2 + ' by approval status', fontsize=12)
    # 1st plot factored by a stroke label:
    plt.subplot(1,3,3)
    df.groupby(label)[num_var3].mean().sort_values(ascending=False).plot(kind='bar')
    plt.xticks(lab_list, lab_name, rotation=0, fontsize=12)
    plt.xlabel('approval status')
    plt.ylabel('Mean ' + num_var3)
    plt.title('Mean ' + num_var3 + ' by approval status', fontsize=12)

def EDA_plot_mean_cat_barchart(df, num_var, cat_var1, cat_var2, cat_var3, cat_var4):
    '''plot 2 by 2 average numerical bar charts categorized by categorical variables'''
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(hspace=1, wspace=1.2)
    # 1st plot
    plt.subplot(2,2,1)
    df.groupby(cat_var1)[num_var].mean().sort_values(ascending=False).plot(kind='bar')
    plt.xlabel(cat_var1)
    plt.ylabel('Mean ' + num_var)
    plt.title('Mean ' + num_var + ' stroke patients by '+ cat_var1)
    # 2nd plot
    plt.subplot(2,2,2)
    df.groupby(cat_var2)[num_var].mean().sort_values(ascending=False).plot(kind='bar')
    plt.xlabel(cat_var2)
    plt.ylabel('Mean ' + num_var)
    plt.title('Mean ' + num_var + ' stroke patients by '+ cat_var2)
    # 3rd plot
    plt.subplot(2,2,3)
    df.groupby(cat_var3)[num_var].mean().sort_values(ascending=False).plot(kind='bar')
    plt.xlabel(cat_var3)
    plt.ylabel('Mean ' + num_var)
    plt.title('Mean ' + num_var + ' stroke patients by '+ cat_var3)
    # 4th plot
    plt.subplot(2,2,4)
    df.groupby(cat_var4)[num_var].mean().sort_values(ascending=False).plot(kind='bar')
    plt.xlabel(cat_var4)
    plt.ylabel('Mean ' + num_var)
    plt.title('Mean ' + num_var + ' stroke patients by '+ cat_var4)   
    
def EDA_plot_hist_label(df, num_var, cat_var, bins, lab_list):
    '''split dataframe by category and plot a histogram'''
    for label in lab_list:
        df_by_label = df[num_var][df[cat_var] == label]
        plt.hist(df_by_label, bins=bins, label=label)
        plt.title('Histogram of ' + str(num_var))
        plt.xlabel(str(num_var))
        plt.ylabel('# of customers')

def EDA_plot_hist_3by1(df, 
                       var1, bin1, lab1, 
                       var2, bin2, lab2, 
                       var3, bin3, lab3,
                       factor=None):
    '''Print skewness and plot the histogram'''
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=1/2, wspace=1/2)
    #subplot 1:
    print("Skewness is:" + lab1, df[var1].skew())
    plt.subplot(3,1,1)
    plt.hist(df[var1]*factor, color='green', bins=bin1)
    plt.title('Histogram of '+ lab1)
    plt.xlabel(lab1)
    plt.ylabel('# of patients')
    #subplot 2:
    print("Skewness is:" + lab2, df[var2].skew())
    plt.subplot(3,1,2)
    plt.hist(df[var2]*factor, color='blue', bins=bin2)
    plt.title('Histogram of '+ lab2)
    plt.xlabel(lab2)
    plt.ylabel('# of patients')
    #subplot 3:
    print("Skewness is:" + lab3, df[var3].skew())
    plt.subplot(3,1,3)
    plt.hist(df[var3]*factor, color='cyan', bins=bin3)
    plt.title('Histogram of '+ lab3)
    plt.xlabel(lab3)
    plt.ylabel('# of patients')

def split_data_by_label(df, label):
    '''set label as an index and split dataframe by an index'''
    df_label = df.set_index(label)
    df_label_1, df_label_0 = df_label.loc[1], df_label.loc[0]
    return (df_label_1, df_label_0)

def split_groupby_data(df,label,cat_var):
    '''Grouped dataframe using a label and categorical varialbe
        then split a dataframe by label(s)'''
    df_grp = pd.DataFrame(df.groupby([label,cat_var])[cat_var].count())
    df_grp.columns = ['count']
    df_grp_0 = df_grp.loc[0]
    df_grp_1 = df_grp.loc[1]
    return(df_grp_0, df_grp_1)

def plot_pie_charts(df_0, df_1, label, var_name, color):
    '''Plot a series of pie charts by a stroke label'''
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    # subplot for non-stroke patients group:
    ax1.pie(df_0, colors=color, autopct='%.0f%%',
            wedgeprops={'edgecolor':'white'}, textprops={'fontsize':14})
    ax1.set_title('Non-Approved Customers ' + 'by ' + var_name, fontsize=13)
    ax1.legend(labels = label, loc='upper right')
    # subplot for stroke patients group:
    ax2.pie(df_1, colors=color, autopct='%.0f%%',
            wedgeprops={'edgecolor':'white'}, textprops={'fontsize':14})
    ax2.set_title('Approved Customers ' + 'by ' + var_name, fontsize=13)
    ax2.legend(labels = label, loc='upper right')
    
def EDA_plot_freq_chart(df, cat_var, var_name):
    '''computes frequency count chart'''
    cat_var_count = df[cat_var].value_counts()
    sns.barplot(cat_var_count.index, cat_var_count.values, alpha=0.9)
    plt.title('Frequency Counts of '+ var_name)
    plt.ylabel('Counts')
    plt.xlabel(var_name, fontsize=10)
    plt.xticks(rotation=270)
    plt.show()

def EDA_plot_bar(cleaned_df, cat_var, num_var, color):
    '''Plots the bar chart'''
    cleaned_df.plot.bar(color=color)
    plt.xlabel(cat_var)
    plt.ylabel('Avg. ' + num_var)
    plt.xticks(rotation=0)
    plt.show()    

def EDA_plot_box_whisker(df, num_var, cat_var, hue=None):
    '''plot the box-whisker plot'''
    df.sort_values(by=[num_var, cat_var], ascending=False, inplace=True)
    plt.figure()
    sns.set(style='whitegrid')
    sns.boxplot(cat_var, num_var, hue, df)
    plt.title('Box Plot of '+ num_var + ' by '+ cat_var)
    plt.xticks(rotation=270, fontsize=9)

def EDA_convert_object_to_cat(df):
    '''convert data type object to category'''
    for col in df.columns:
        if df[col].dtype.name == "object":
            df[col] = df[col].astype('category')

def EDA_encode_cat_var(df, col, num_var):
    '''encode the categorical variables using a specified numerical variable for each category '''
    cat_dict={}
    cats = df[col].cat.categories.tolist()
    for cat in cats:
        cat_dict[cat] = df[df[col] == cat][num_var].mean()
    df[col] = df[col].map(cat_dict)

def EDA_plot_corr_matrix(df, features, label):
    '''plot the correlation matrix'''
    corr = df[features + label].corr()
    # Create a mask:
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(12,10))
    sns.heatmap(corr,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                annot=True, fmt=".2f", mask=mask)
    plt.xticks(rotation=90)
    plt.show()

def EDA_plot_crosstab(df, cat_var1, cat_var2):
    '''plot a cross-tabulate on two categorical variables'''
    cross_tab = pd.crosstab(df[cat_var1], df[cat_var2])
    return cross_tab

def EDA_plot_scatter(df, 
                     var1, lab1, c1,
                     var2, lab2, c2, 
                     factor=None):
    '''plot 2 by 1 scatter plots'''
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=0.4, wspace=0.9)
    plt.subplot(2,1,1)
    plt.scatter(df['age'], df[var1]*factor, color=c1)
    plt.title('Relationship between ' + lab1 + ' and Age')
    plt.xlabel('Age')
    plt.ylabel(lab1)
    
    plt.subplot(2,1,2)
    plt.scatter(df['age'], df[var2]*factor, color=c2)
    plt.title('Relationship between '+ lab2 + ' and Age')
    plt.xlabel('Age')
    plt.ylabel(lab2)
    
def Pearson_r(df, x, y):
    '''compute Pearson r correlation'''
    corr_mat = np.corrcoef(df[x],df[y])
    return corr_mat[0, 1]

def EDA_plot_pair_linreg(df_0, df_1, num_var1, num_var2, 
                         r_0, r_1, lab0, lab1):
    '''plot a pair of linear regressions'''
    plt.figure(figsize=(10,5))
    plt.subplots_adjust(wspace=0.7)
    # plot 1:
    plt.subplot(1,2,1)
    plt.plot(df_1[num_var1], df_1[num_var2], 'r--', label='r =%.2f' % r_1,
             marker='.', linestyle='none', color='red')
    plt.margins(0.02)
    plt.legend(loc='upper left')
    plt.xlabel(num_var1)
    plt.ylabel(num_var2)
    plt.title(num_var1 + ' vs. ' + num_var2 + ' by ' + lab1)
    # Fit linear regression:
    a,b = np.polyfit(df_1[num_var1], df_1[num_var2], 1)
    x = np.array([0, 84])
    y = a*x + b
    plt.plot(x,y)
    
    # plot 2:
    plt.subplot(1,2,2)
    plt.plot(df_0[num_var1], df_0[num_var2], 'g--', label='r =%.2f' % r_0,
             marker='.', linestyle='none', color='green')
    plt.margins(0.02)
    plt.legend(loc='upper left')
    plt.xlabel(num_var1)
    plt.ylabel(num_var2)
    plt.title(num_var1 + ' vs. ' + num_var2 + ' by ' + lab0)
    # Fit linear regression:
    a,b = np.polyfit(df_0[num_var1], df_0[num_var2], 1)
    x = np.array([0, 84])
    y = a*x + b
    plt.plot(x,y)
    
def EDA_plot_multi_facet_scatter(df1, df2, 
                                 var1, lab1, 
                                 var2, lab2,
                                 response, factor):
    '''plot multi-faceted scatter plot by county class'''
    f, (ax1, ax2)=plt.subplots(1, 2, sharey=True, figsize=(8,4))
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
    ax1.scatter(df1[var1]*factor, df1[response], label='Nonmetro', edgecolor='w')
    ax1.scatter(df2[var1]*factor, df2[response], label='Metro', edgecolor='w')
    ax1.legend(loc='upper right')
    ax1.set_xlabel(lab1, fontsize=10)
    ax1.set_ylabel(response, fontsize=10)
    ax1.grid(False)
    
    ax2.scatter(df1[var2]*factor, df1[response], label='Nonmetro', edgecolor='w')
    ax2.scatter(df2[var2]*factor, df2[response], label='Metro', edgecolor='w')
    ax2.legend(loc='upper right')
    ax2.set_xlabel(lab2, fontsize=10)
    ax2.set_ylabel(response, fontsize=10)
    ax2.grid(False)

def EDA_plot_color_sc_scatter(df, var1, lab1,
                              var2, lab2, response):
    '''plot color scaled scatter plots'''
    # figure 1: subplot 1
    f, (ax1) = plt.subplots(1,1, sharey=True, figsize=(10,6))
    s1 = ax1.scatter(df[var1], df[var2], c=df[response],
                     cmap=plt.cm.coolwarm, edgecolor='w')
    ax1.set_xlabel(lab1, fontsize=14)
    ax1.set_ylabel(lab2, fontsize=14)
    ax1.grid(False)
    # lenged: color bar scaled by confounding factor
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax=plt.axes([0.85, 0.1, 0.05, 0.8])
    cb=f.colorbar(s1, cax=cax)
    cb.set_label(response)
    
# --- 3. Load the data --- #
# Define input CSVs:
if __name__ == '__main__':
    EDA_file = 'df_EDA.csv'

# Load data
df_EDA = load_file(EDA_file)

# Define type of variables list:
#df_EDA.select_dtypes(include='object').columns
cat_vars = ['Gender', 'City_Code', 'City_Category', 'Employer_Code',
       'Employer_Category1', 'Employer_Category2',
       'Customer_Existing_Primary_Bank_Code', 'Primary_Bank_Type', 'Contacted',
       'Source', 'Source_Category', 'Var1']

# Define type of variables list:
#df_EDA.select_dtypes(include='int64').columns
#df_EDA.select_dtypes(include='float64').columns
num_con_vars = ['Monthly_Income', 'Existing_EMI', 'Loan_Amount', 'Loan_Period',
       'Interest_Rate', 'EMI', 'Age']

num_disc_vars = ['LCD_year', 'LCD_month', 'LCD_day', 'LCD_weekday',
       'Is_Missing_Loan_Amount', 'Is_Missing_EMI', 'Is_Missing_Interest_Rate']
# 'ID'

# Define variables to drop
lab_var = ['Approved']
################################################
# Part 3 - Exploratory Data Analysis: Insights #
################################################
# compute top 10 rows on a eda_dataframe:
df_EDA.head()    

# check duplicates:
#df_clean = clean_data(df_EDA)

#---- Compute % of Missing Data ----#
missing_df = EDA_missing_data(df_EDA)
missing_df
    
#---- Compute Summary Statistics: numerical data ----#
df_stat_num = EDA_numerical_data(df_EDA[num_con_vars])
df_stat_num

#---- Compute Summary Statistics: categorical data ----#
df_stat_cat = EDA_categorical_data(df_EDA)
df_stat_cat

#---- Visualize response variable (salary) ----#
EDA_plot_num_var(df_EDA, 'Age')

EDA_plot_num_var(df_EDA, 'Loan_Amount')

EDA_plot_num_var(df_EDA, 'Loan_Period')

#--- Use IQR to detect potential outliers ----#
stat = df_EDA.Age.describe()
print(stat)
IQR = stat['75%'] - stat['25%']
UB = stat['75%'] + 1.5*IQR
LB = stat['25%'] - 1.5*IQR
print('The LB and UB for suspected Age outliers are {} and {}. '.format(LB, UB))

# Check LB Outeliers:
df_EDA[df_EDA.Age < 17]

# Check UB Outeliers:
df_EDA[df_EDA.Age > 49]

# check potential outliers by categorical vars:
df_EDA.loc[df_EDA.Age > 49, 'Primary_Bank_Type'].value_counts()

#--- Check the suspicious outliers by 

# Create a separate dataframe by label:
df_approved, df_non_approved = split_data_by_label(df_EDA, 'Approved')

#############################
#---- Plot bar chart(s) ----#
#############################
# Create a list of label:
lab_list = list(df_EDA['Approved'].unique())
lab_name = ['Non-Approved', 'Approved']

EDA_plot_mean_label_barchart(df_EDA, 'Age', 'Monthly_Income', 'Loan_Amount',
                             'Approved', lab_list, lab_name)

EDA_plot_mean_cat_barchart(df_EDA, 'Loan_Amount', 'Gender', 'City_Category',
                             'Employer_Category2', 'Source_Category')
###########################
#---- Plot pie charts ----#
###########################
# Split a groupby dataframe by Gender:
df_grp_gender_0, df_grp_gender_1 = split_groupby_data(df_EDA, 'Approved', 'Gender')
# Split a groupby dataframe by Contacted:
df_grp_contacted_0, df_grp_contacted_1 = split_groupby_data(df_EDA, 'Approved', 'Contacted')
# Split a groupby dataframe by Contacted:
df_grp_bank_type_0, df_grp_bank_type_1 = split_groupby_data(df_EDA, 'Approved', 'Primary_Bank_Type')

# Plot a set of pie charts: by Gender
plot_pie_charts(df_grp_gender_0, df_grp_gender_1, df_grp_gender_0.index,
                'Gender', ['red', 'green'])

plot_pie_charts(df_grp_contacted_0, df_grp_contacted_1, df_grp_contacted_0.index,
                'Contacted', ['red', 'green'])

plot_pie_charts(df_grp_bank_type_0, df_grp_bank_type_1, df_grp_bank_type_0.index,
                'Primary_Bank_Type', ['deepskyblue','lawngreen','orange'])

###########################
#---- Plot histograms ----#
###########################
# Plot multiple histograms on numerical variables by a label
EDA_plot_hist_label(df_EDA, 'Age', 'Approved', 20, lab_list)
plt.legend(('Non-Approved', 'Approved'), loc='upper_right')
plt.show()

EDA_plot_hist_label(df_EDA, 'Loan_Amount', 'Approved', 20, lab_list)
plt.legend(('Non-Approved', 'Approved'), loc='upper_right')
plt.show()

EDA_plot_hist_label(df_EDA, 'Loan_Period', 'Approved', 20, lab_list)
plt.legend(('Non-Approved', 'Approved'), loc='upper_right')
plt.show()

EDA_plot_hist_label(df_EDA, 'Interest_Rate', 'Approved', 20, lab_list)
plt.legend(('Non-Approved', 'Approved'), loc='upper_right')
plt.show()

# Plot multiple histograms:
# Plot 3by1 histogram as a subplot: 
EDA_plot_hist_3by1(df_EDA, 'Monthly_Income', 15, 'Monthly Income',
                           'Existing_EMI', 15, 'Existing EMI',
                           'EMI', 15, 'EMI', 1)    

EDA_plot_hist_3by1(df_EDA, 'Loan_Amount', 15, 'Loan Amount',
                           'Loan_Period', 15, 'Loan Period',
                           'Interest_Rate', 15, 'Interest Rate', 1)    


###################################
#---- Plot frequency chart(s) ----#
###################################
# plot frequency chart: 
EDA_plot_freq_chart(df_EDA, 'Employer_Category1', 'Employer Category 1')

EDA_plot_freq_chart(df_EDA, 'Employer_Category2', 'Employer Category 2')

EDA_plot_freq_chart(df_EDA, 'Primary_Bank_Type', 'Primary Bank Type')

EDA_plot_freq_chart(df_EDA, 'Source_Category', 'Source Category')

##########################################
#---- Plot box-whisker plot chart(s) ----#
##########################################
# Plot box plot" 
EDA_plot_box_whisker(df_EDA, 'Age', 'Primary_Bank_Type')

EDA_plot_box_whisker(df_EDA, 'Age', 'Employer_Category1')

EDA_plot_box_whisker(df_EDA, 'Age', 'Source_Category')

#---- Convert categorical variable data type from object to category ----#
df_EDA_cm = df_EDA.copy()
EDA_convert_object_to_cat(df_EDA_cm)

#---- Encode categorical variables using avg. salary for each category to replace label ----#
for col in df_EDA.columns:
    if df_EDA_cm[col].dtype.name == 'category':
        EDA_encode_cat_var(df_EDA_cm, col, 'Monthly_Income')

#---- Plot correlation matrix chart ----#
# Define list of features and Monthly_Income
features = ['Existing_EMI', 'Loan_Amount', 'Interest_Rate', 'EMI', 'Age', 'Gender',
            'City_Code', 'City_Category', 'Employer_Category1',
            'Employer_Category2', 'Customer_Existing_Primary_Bank_Code',
            'Primary_Bank_Type', 'Contacted', 'Source', 'Source_Category', 'Var1']
label = ['Approved']

EDA_plot_corr_matrix(df_EDA_cm, features, label)
########################################################
# Bi-variate analyses: cross-tabulation, scatter plots #
########################################################
#---- Plot a cross-tabulate based on two categorical variables ----#    
EDA_plot_crosstab(df_EDA, 'Source_Category', 'Employer_Category1')

EDA_plot_crosstab(df_EDA, 'City_Category', 'Primary_Bank_Type')

#---- Plot a scatter plot: numerical and categorical variables ----#

#---- Plot multi-faceted scatter plots by categorical variable ----#

#---- Plot color scaled scatter plots by numerical variable ----#
EDA_plot_color_sc_scatter(df_EDA, 'Interest_Rate','Interest Rate',
                          'EMI', 'EMI', 'Age')
plt.show()