#imports
import pandas as pd
import numpy as np
import datetime
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from collections import Counter
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split



#handle missing data
def Missing_values(df):
    #null values
    print ("summary of null values befor imputing the data: ")
    print(df.isna().sum())
    #impute data by mode
    return df.groupby(['device_width','device_height','device_version']).apply(lambda x: x.fillna(x.mode().iloc[0])).reset_index(drop=True)


# create new  features
def Feature_engineering(df):
    # hours and month dictionaries
    hour_dict = {0: 'evening', 1: 'evening',
                 2: 'night', 3: 'night', 4: 'night', 5: 'night',
                 6: 'night', 7: 'night', 8: 'night', 9: 'night', 10: 'night',
                 11: "day", 12: "day", 13: "day", 14: "day",
                 15: "day", 16: "day", 17: "day",
                 18: "evening", 19: "evening", 20: "evening",
                 21: "evening", 22: "evening", 23: "evening"}

    month_dict = {'Nov': 'fall', 'Oct': 'fall', 'Sep': 'fall',
                  'Aug': 'summer', 'Sep': 'summer'}
    # decive diagonal
    df.loc[:, 'device_diag'] = np.sqrt(df.device_height ^ 2 * df.device_width ^ 2).round()
    # time instead timestamp
    df.loc[:, 'time'] = pd.to_datetime(df['timestamp'], unit='s')
    # day of the week
    df.loc[:, 'Day_of_Week'] = df['time'].dt.weekday_name
    # month and season
    df.loc[:, 'Month'] = df['time'].dt.month_name().str[:3].map(month_dict)
    # hour and day time
    df.loc[:, 'hour'] = df['time'].dt.hour.map(hour_dict)
    # drop time and timestamp
    df = df.drop(['time', 'timestamp'], axis=1)

    return df


# normalize the data
def Data_normalization(df, method='z'):
    # the data is numeric
    if len(df.columns) - len(df._get_numeric_data().columns) == 0:
        # normalization between -1 and 1
        if method == 'z':
            # create a scaler object
            std_scaler = StandardScaler()
            # fit and transform the data
            df_std = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)
            return df_std
        # normalization between 0 and 1
        if method == '0-1':
            x = df.values  # returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df_min_max = pd.DataFrame(x_scaled)
            return df_min_max

    else:
        print("function gets only int values")


# from categories data into boolean data
def get_dummies_fun(df, cat_list):
    df_copy = df
    df_dummies = pd.get_dummies(df_copy, prefix='', prefix_sep='',
                                columns=cat_list)
    return df_dummies




def cumulatively_categorise_f(column,threshold=0.85,return_categories_list=False):
  #Find the threshold value using the percentage and number of instances in the column
    threshold_value=int(threshold*len(column))
  #Initialise an empty list for our new minimised categories
    categories_list=[]
  #Initialise a variable to calculate the sum of frequencies
    s=0
  #Create a counter dictionary of the form unique_value: frequency
    counts=Counter(column)

  #Loop through the category name and its corresponding frequency after sorting the categories by descending order of frequency
    for i,j in counts.most_common():
    #Add the frequency to the global sum
        s+=dict(counts)[i]
    #Append the category name to the list
        categories_list.append(i)
    #Check if the global sum has reached the threshold value, if so break the loop
        if s>=threshold_value:
            break
  #Append the category Other to the list
    categories_list.append('Other')

  #Replace all instances not in our new categories by Other
    new_column=column.apply(lambda x: x if x in categories_list else 'Other')

  #Return transformed column and unique values if return_categories=True
    if(return_categories_list):
        return new_column,categories_list
  #Return only the transformed column if return_categories=False
    else:
        return new_column



#Reducing the amount of categories
def cumulatively_categorise(df,columns):
    for col in columns:
        df[col] =cumulatively_categorise_f(df[col])
    return df


# handle imbalanced data
def Smote_alg(X, y):
    # transform the dataset
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    return X, y


