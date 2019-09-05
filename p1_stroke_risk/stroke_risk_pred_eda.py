# Import and set directory:
import os
os.getcwd()

# Set path:
#path = 'D:/P_Poverty_Prediction/A. Coding'
path = 'D:/TY Workspace/3. Data Science Portfolio/1. Completed Portfolios/p1_stroke_risk/A. Coding'
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
    return pd.read_csv(file)

def clean_data(raw_df):
    '''remove rows that contain invalid data or duplicate IDs'''
    clean_df = raw_df.drop_duplicates(subset='id')
    return clean_df

def EDA_missing_data(cleaned_df):
    '''Performs missing % on each column '''
    missing_df = cleaned_df.isnull().sum()
    missing_df = pd.DataFrame(missing_df, columns=['count'])
    missing_df['pct'] = missing_df['count']/len(cleaned_df)
    missing_df = missing_df.sort_values(by='pct', ascending=False)
    return missing_df

def EDA_numerical_data(cleaned_df):
    '''Computes summary statistics on numerical data'''
    summary_df_num = cleaned_df.describe(include='all').T
    summary_df_num = pd.DataFrame(summary_df_num)[['count', 'std', 'min', 'mean', 'max', '25%', '50%', '75%']]
    return summary_df_num

def EDA_categorical_data(cleaned_df):
    '''Computes summary statitics on categorical data'''
    summary_df_cat = cleaned_df.describe(include=['O'])
    summary_df_cat = pd.DataFrame(summary_df_cat)
    return summary_df_cat
    
def EDA_pivot_table(cleaned_df, cat_var, num_var):
    '''Creates a pivot table based on categorical var and average numerical var'''
    pivot_cat_df = cleaned_df.pivot_table(index=cat_var, values=num_var, aggfunc=np.mean)
    pivot_cat_df.reset_index(level=0, inplace=True)
    pivot_cat_df.rename(columns={cat_var:cat_var, num_var:"avg" + "_" + num_var + "_" + cat_var}, inplace=True)
    return pivot_cat_df
    
def EDA_plot_hist_label(df, num_var, cat_var, bins, lab_list):
    '''split dataframe by category and plot a histogram'''
    for i in lab_list:
        df_by_label = df[num_var][df[cat_var] == i]
        plt.hist(df_by_label, bins=bins, label=i)
        plt.title('Histogram of ' + str(num_var))
        plt.xlabel(str(num_var))
        plt.ylabel('# of patients')                   

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

def split_groupby_data(df,label,cat_var):
    '''Grouped dataframe using a label and categorical varialbe
        then split a dataframe by label(s)'''
    df_grp = pd.DataFrame(df.groupby([label,cat_var])[cat_var].count())
    df_grp.columns = ['count']
    df_grp_0 = df_grp.loc[0]
    df_grp_1 = df_grp.loc[1]
    return(df_grp_0, df_grp_1)

def plot_pie_charts(df_grp_0, df_grp_1, label, var_name, color):
    '''Plot a set of pie charts for non-stroke/stroke cases'''
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    # subplot for non-stroke:
    ax1.pie(df_grp_0, colors=color, autopct='%.0f%%',
            wedgeprops={'edgecolor':'white'}, textprops={'fontsize':14})
    ax1.set_title('Non-stroke patients ' + var_name, fontsize=13)
    ax1.legend(labels = label, loc='upper right')
    # subplot for stroke:
    ax2.pie(df_grp_1, colors=color, autopct='%.0f%%',
            wedgeprops={'edgecolor':'white'}, textprops={'fontsize':14})
    ax2.set_title('Stroke patients ' + var_name, fontsize=13)
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
    plt.scatter(df[var1]*factor, df['age'], color=c1)
    plt.title('Relationship between ' + lab1 + ' and Age')
    plt.xlabel(lab1)
    plt.ylabel('Age')

    plt.subplot(2,1,2)
    plt.scatter(df[var2]*factor, df['age'], color=c2)
    plt.title('Relationship between '+ lab2 + ' and Age')
    plt.xlabel(lab2)
    plt.ylabel('Age')
    
def convert_data_type_category(df, var_name):
    df_eda[var_name] = df_eda[var_name].astype('str')
    return df_eda[var_name]
    
def split_dataframe_by_string(df, cat_var, str_val):
    '''split dataframe by a specified string value in categorical variable'''
    df_str = df[df[cat_var].str.contains(str_val, case=True, regex=False)]
    return df_str

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
    eda_file = 'df_eda_stroke.csv'

# Define type of variables list:
#df_eda.select_dtypes(include='object').columns
cat_vars = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
            'work_type', 'Residence_type', 'smoking_status']
#df_eda.select_dtypes(include='int64').columns
#df_eda.select_dtypes(include='float64').columns
num_vars = ['age', 'avg_glucose_level', 'bmi']
label_var = 'stroke'

# Define variable(s) to drop:
list_vars =['id']

# Load data
df_eda = load_file(eda_file)

################################################
# Part 3 - Exploratory Data Analysis: Insights #
################################################
# compute top 10 rows on a eda_dataframe:
df_eda.head(10)

# check duplicates:
df_eda.duplicated().sum()

#---- Compute % of Missing Data ----#
missing_df = EDA_missing_data(df_eda)
missing_df
    
#---- Compute Summary Statistics: numerical data ----#
summary_df_num = EDA_numerical_data(df_eda[num_vars])
summary_df_num

#---- Compute Summary Statistics: categorical data ----#
summary_df_cat = EDA_categorical_data(df_eda[cat_vars])
summary_df_cat

#---- Visualize numerical variable (age) ----#
plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
sns.boxplot(df_eda.age)
plt.subplot(1,2,2)
sns.distplot(df_eda.age, bins=20)
plt.show()

#---- Visualize numerical variable (avg_glucose_level) ----#
plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
sns.boxplot(df_eda.avg_glucose_level)
plt.subplot(1,2,2)
sns.distplot(df_eda.avg_glucose_level, bins=20)
plt.show()

#--- Use IQR to detect potential outliers ----#
stat = df_eda.avg_glucose_level.describe()
print(stat)
IQR = stat['75%'] - stat['25%'] 
UB = stat['75%'] + 1.5*IQR
LB = stat['25%'] - 1.5*IQR
print('The LB and UB for suspected avg. glucose outliers are {} and {}. '.format(LB, UB))

# Check LB Outeliers:
df_eda[df_eda.avg_glucose_level < 25.745]

# Check UB Outeliers:
df_eda[df_eda.avg_glucose_level > 163.8649]

# check potential outliers by categorical vars:
df_eda.loc[df_eda.avg_glucose_level > 163.8649, 'work_type'].value_counts()
df_eda.loc[df_eda.avg_glucose_level > 163.8649, 'smoking_status'].value_counts()

#--- Check the suspicious outliers by an economic typology: mining-dependent
df_eda[(df_eda.avg_glucose_level > 163.8649) & (df_eda.smoking_status == 'never smoked')]

#---- Plot pie charts ----#
# Split a groupby dataframe by stroke label: gender
df_grp_gender_0, df_grp_gender_1 = split_groupby_data(df_eda,'stroke','gender')
# Split a groupby dataframe by stroke label: marital_status
df_grp_marital_0, df_grp_marital_1 = split_groupby_data(df_eda,'stroke','ever_married')
# Split a groupby dataframe by stroke label: smoking_status
df_grp_smoking_0, df_grp_smoking_1 = split_groupby_data(df_eda,'stroke','smoking_status')
# Split a groupby dataframe by stroke label: hypertension
df_grp_hypertension_0, df_grp_hypertension_1 = split_groupby_data(df_eda,'stroke','hypertension')
# Split a groupby dataframe by stroke label: heart_disease
df_grp_heart_disease_0, df_grp_heart_disease_1 = split_groupby_data(df_eda,'stroke','heart_disease')

# Plot a set of pie charts: by gender
plot_pie_charts(df_grp_gender_0, df_grp_gender_1, df_grp_gender_0.index,
                'gender', ['red', 'blue', 'purple'])
# Plot a set of pie charts: by marital_status
plot_pie_charts(df_grp_marital_0, df_grp_marital_1, df_grp_marital_0.index,
                'marital status', ['red', 'green'])
# Plot a set of pie charts: by smoking_status
plot_pie_charts(df_grp_smoking_0, df_grp_smoking_1, df_grp_smoking_0.index,
                'smoking status', ['deepskyblue','lawngreen','orange'])
# Plot a set of pie charts: by hypertension
plot_pie_charts(df_grp_hypertension_0, df_grp_hypertension_1, df_grp_hypertension_0.index,
                'hypertension', ['green', 'red'])
# Plot a set of pie charts: by heart_disease
plot_pie_charts(df_grp_heart_disease_0, df_grp_heart_disease_1, df_grp_heart_disease_0.index,
                'heart disease', ['green', 'red'])


#---- Plot histograms ----#
# Create a list of economic typology:
lab_list = list(df_eda['stroke'].unique())                 

# Plot multiple histograms on numerical variables by a stroke label:
EDA_plot_hist_label(df_eda, 'avg_glucose_level', 'stroke', 20, lab_list)
plt.legend(('non-stroke','stroke'), loc='upper right')
plt.show()

EDA_plot_hist_label(df_eda, 'age', 'stroke', 20, lab_list)
plt.legend(('non-stroke','stroke'), loc='upper right')
plt.show()

EDA_plot_hist_label(df_eda, 'bmi', 'stroke', 20, lab_list)
plt.legend(('non-stroke','stroke'), loc='upper right')
plt.show()
    
# Plot 3by1 histogram as a subplot: numerical variables
EDA_plot_hist_3by1(df_eda,
                   'avg_glucose_level', 15, 'Avg. Glucose Level',
                   'bmi', 15, 'Body Mass Index',
                   'age', 15, 'Age', 1)
plt.show()

#---- Plot bar chart(s) ----#
# Plot bar chart: smoking_status
EDA_plot_freq_chart(df_eda, 'smoking_status', 'Smoking Status')
plt.show()

# Plot bar chart: work_type
EDA_plot_freq_chart(df_eda, 'work_type', 'Work Type')
plt.show()

# Plot bar chart: residence_type
EDA_plot_freq_chart(df_eda, 'Residence_type', 'Residence Type')
plt.show()

# Plot bar chart: gender
EDA_plot_freq_chart(df_eda, 'gender', 'Gender')
plt.show()

#---- Plot box-whisker plot chart(s) ----#
# Plot box plot: Smoking Status
EDA_plot_box_whisker(df_eda, 'age', 'smoking_status')
plt.show()
# Plot box plot: Work Type
EDA_plot_box_whisker(df_eda, 'age', 'work_type')
plt.show()
# Plot box plot: Residence Type
EDA_plot_box_whisker(df_eda, 'age', 'Residence_type')
plt.show()

# Drop 'row_id':
#df_eda.drop(list_vars, axis=1, inplace=True)

#---- Convert categorical variable data type from object to category ----#
df_eda_cm = df_eda.copy() 
EDA_convert_object_to_cat(df_eda_cm)

#---- Encode categorical variables using avg. numerical variable for each category to replace label ----#
for col in df_eda_cm.columns:
    if df_eda_cm[col].dtype.name == "category":
        EDA_encode_cat_var(df_eda_cm, col, 'age')
        
#---- Plot correlation matrix chart ----#
# Define list of features and salary
features = ['age', 'avg_glucose_level', 'bmi', 'gender',
            'hypertension', 'heart_disease',  'ever_married', 
            'work_type', 'Residence_type',  'smoking_status']
label = ['stroke']

# Plot a correlation matrix:
EDA_plot_corr_matrix(df_eda_cm, features, label)

########################################################
# Bi-variate analyses: cross-tabulation, scatter plots #
########################################################
#---- Plot a cross-tabulate based on two categorical variables ----#    
EDA_plot_crosstab(df_eda, 'heart_disease', 'hypertension')

EDA_plot_crosstab(df_eda, 'work_type', 'Residence_type')

#---- Plot a scatter plot: numerical and categorical variables ----#
# Demographics
EDA_plot_scatter(df_eda, 
                 'avg_glucose_level', 'Avg Glucose Level', 'green',
                 'bmi', 'Body Mass Index', 'blue', 1)
plt.show()

#---- Plot multi-faceted scatter plots by categorical variable ----#

#---- Plot color scaled scatter plots by numerical variable ----#
EDA_plot_color_sc_scatter(df_eda, 'avg_glucose_level','Avg Glucose Level',
                          'bmi', 'Body Mass Index', 'age')

plt.show()