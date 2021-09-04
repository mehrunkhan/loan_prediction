import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# load data to pandas dataframe
raw_data = pd.read_csv("../data/dataset_after_column_selection.csv", index_col=None)

raw_data.rename(
    columns=({'Document_Visa/Resident permit': 'Document_Visa_or_Resident_permit'}),
    inplace=True,
)
# There are some XNA values in CODE_GENDER column, we replace them with F
raw_data["CODE_GENDER"].replace({"XNA": "F"}, inplace=True)

# In NAME_FAMILY_STATUS,NAME_HOUSING_TYPE and NAME_EDUCATION_TYPE's column value we replace '/' with 'or'
raw_data.NAME_FAMILY_STATUS = raw_data.NAME_FAMILY_STATUS.apply(
    lambda x: x.replace('/', 'or'))
raw_data.NAME_HOUSING_TYPE = raw_data.NAME_HOUSING_TYPE.apply(
    lambda x: x.replace('/', 'or'))
raw_data.NAME_EDUCATION_TYPE = raw_data.NAME_EDUCATION_TYPE.apply(
    lambda x: x.replace('/', 'or'))

# Drop non relevant columns from the main dataframe
raw_data.drop('Document_University_Info', inplace=True, axis=1)
raw_data.drop('SK_ID_CURR', inplace=True, axis=1)
raw_data.drop('ORGANIZATION_TYPE', inplace=True, axis=1)
raw_data.drop('FLAG_MOBIL', inplace=True, axis=1)
raw_data.drop('DAYS_EMPLOYED', inplace=True, axis=1)
raw_data.drop('REGION_RATING_CLIENT', inplace=True, axis=1)
raw_data.drop('REG_REGION_NOT_WORK_REGION', inplace=True, axis=1)


# Multiply 10 with AMT_INCOME_TOTAL column for synchronizing it with other column's value
raw_data["AMT_INCOME_TOTAL"] = 10 * raw_data["AMT_INCOME_TOTAL"]

# Filling up some null values of OCCUPATION_TYPE column with a specific value
raw_data["OCCUPATION_TYPE"].fillna("Production_Worker", inplace=True)

# Alternation some values of OCCUPATION_TYPE column for making this column more relevant
raw_data.OCCUPATION_TYPE = raw_data.OCCUPATION_TYPE.replace(
    {"Realty agents": "restaurant worker",
     "Managers": "restaurant worker", "Core staff": "restaurant worker"})

# Fill up all null values of DAYS_LAST_PHONE_CHANGE column  with 0
raw_data["DAYS_LAST_PHONE_CHANGE"].fillna(0, inplace=True)

# Fill up all null values of CNT_FAM_MEMBERS column  with 2
raw_data["CNT_FAM_MEMBERS"].fillna(2, inplace=True)

# Turns CNT_FAM_MEMBERS column from float to int
raw_data["CNT_FAM_MEMBERS"] = raw_data["CNT_FAM_MEMBERS"].astype(int)

# Reorganize AMT_INCOME_TOTAL column with a relevant range
raw_data['AMT_INCOME_TOTAL'] = raw_data['AMT_INCOME_TOTAL'].mask(
    raw_data['AMT_INCOME_TOTAL'] > 3500, 1550)

# Switch the value of CNT_FAM_MEMBERS column from greater than 5 to 4
raw_data['CNT_FAM_MEMBERS'] = raw_data['CNT_FAM_MEMBERS'].mask(
    raw_data['CNT_FAM_MEMBERS'] > 5, 4)

# convert the value of CNT_CHILDREN column which are greater than 5 to 3
raw_data['CNT_CHILDREN'] = raw_data['CNT_CHILDREN'].mask(
    raw_data['CNT_CHILDREN'] > 5, 3)


# All columns name is turned from upper case to lower case, and here we call a function named clean_data
def clean_data():
    column_name = raw_data.columns
    column_name = [i.lower() for i in column_name]
    raw_data.columns = column_name
    return raw_data

print()

