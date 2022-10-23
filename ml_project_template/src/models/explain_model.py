import shap	
import joblib
import pandas as pd
import os
import sys
import numpy as np
import logging
import lxml
from src.features.custom_classes import CustomOneHot, CustomStandardScaler
from src.models.model_utils import get_features
from src.models.model_utils import fillna_data, new_category_check
from src.data.get_data_from_db import get_list_df
from operator import itemgetter as it
from itertools import repeat
import matplotlib.pyplot as plt

# Import user generated modules
from src.data.get_preprocessed_data import preprocessing_data
from src.models.model_utils import fillna_data, get_features, new_category_check

# Setup
user_name = os.environ.get("USERNAME")
model = "rf"
model_version_date = "02-11-2021-BASELINE-v0-6"
data_dump_date = "02-11-2021"
get_data_dumps = False
open_opps_only = False
run_mode = "predict"

# Define file paths
train_data_path = r"C:\Users\{user_name}\Novozymes A S\Customer 360 - Documents\FrozenDatabases\ModelResults\data_files\clean_data\03-05-2022".format(user_name = user_name)
test_data_path = r"C:\Users\PETP\OneDrive - Novozymes A S\Desktop"
pd.options.display.float_format = '{:,.2f}'.format

def get_grp_shap(df, X_train, X_test, cat_dict):
"""
Function to get grouped Shap values for features
Args:
df: the dataset after transforming/scaling used for Shap values calculation
X_train: dataset used for building Shap explainer
"""
categorical_cols, boolean_cols, numerical_cols = get_features(model_version_date)

# Get list of columns used in the model
req_model_cols = numerical_cols + boolean_cols + categorical_cols
model_path = (
r"models/trained_models/"
+ model_version_date
+ r"/{application}.pkl".format(application=model)
)
prod_model= joblib.load(model_path)
trf = prod_model["transformation"]
orig_trf_train = trf.transform(X_test[req_model_cols])
X_train_trf = trf.transform(X_train[req_model_cols])
req_trf_cols = orig_trf_train.columns
df = df[req_trf_cols]
print(df.shape)
nsamples = 4000

# Create object that can calculate shap values based on the train set
explainer = shap.TreeExplainer(prod_model["model"].best_estimator_,
shap.sample(X_train_trf,
nsamples = nsamples,random_state=42),
model_output='probability',)
tree_shap_values = explainer.shap_values(df,
approximate=True,
check_additivity=True,
)
tree_shap_values_arr = np.array(tree_shap_values)
won_shap_df = pd.DataFrame(data = tree_shap_values_arr[1,:,:],columns = req_trf_cols)
lost_shap_df = pd.DataFrame(data = tree_shap_values_arr[0,:,:],columns = req_trf_cols) 
grp_won_shap_df = pd.DataFrame(columns=req_model_cols)
grp_lost_shap_df = pd.DataFrame(columns=req_model_cols)
for col in req_model_cols:
if col not in categorical_cols:
grp_won_shap_df.loc[:,col] = won_shap_df.loc[:,col]
grp_lost_shap_df.loc[:,col] = lost_shap_df.loc[:,col]
else:
grp_won_shap_df.loc[:,col] = won_shap_df.loc[:,cat_dict[col]].sum(axis=1)
grp_lost_shap_df.loc[:,col] = lost_shap_df.loc[:,cat_dict[col]].sum(axis=1)
return tree_shap_values_arr,grp_won_shap_df, grp_lost_shap_df
# Define function for 
def format_time_to_months(id):
'''
Function for formatting time intervals in months and days.
'''
og_value = 0
opp_len = id
if opp_len != np.nan:
days = int((opp_len % 1)*31)
if abs(opp_len) > 1:
if abs(days) > 1:
og_value = str(opp_len).split('.')[0] + " months " + str(days) + " days"
elif days == 1:
og_value = str(opp_len).split('.')[0] + " months " + str(days) + " day"
elif days == 0:
og_value = str(opp_len).split('.')[0] + " months "
elif opp_len == 1:
if days > 1:
og_value = str(opp_len).split('.')[0] + " month " + str(days) + " days"
elif days == 1:
og_value = str(opp_len).split('.')[0] + " month " + str(days) + " day"
else:
og_value = str(opp_len).split('.')[0] + " month"
else:
if days > 1:
og_value = str(days) + " days"
elif days == 1:
og_value = str(days) + " day"
else:
og_value = str(days) + " day"
else:
opp_len = np.nan
return og_value
def format_time_to_years(id):
'''
Function for formatting time intervals in years and months.
'''
og_value = 0
opp_len = id
if opp_len != np.nan:
years = int(opp_len/365)
months = int((opp_len % 365)/30)
if abs(years) > 1:
if abs(months) > 1:
og_value = str(years) + " years " + str(months) + " months"
elif months == 1:
og_value = str(years) + " years " + str(months) + " month"
elif months == 0:
og_value = str(years) + " years "
elif opp_len == 1:
if months > 1:
og_value = str(years) + " year " + str(months) + " months"
elif months == 1:
og_value = str(years) + " year " + str(months) + " month"
else:
og_value = str(years) + " year"
else:
if abs(months) > 1:
og_value = str(months) + " months"
elif months == 1:
og_value = str(months) + " month"
else:
og_value = str(months) + " month"
else:
opp_len=np.nan
return og_value
def explain_model(
model_version_date: str,
data_dump_date: str,
get_data_dumps: bool = False,
threshold: float = 0.7,
nlargest: int = 6,
):
"""
Function for creating output for OWR Model explainability, based on Shap values.
Args:
model_version_date: Model Version date/ Name
data_dump_date: date of data dump
get_data_dumps: Flag, when set to true fetches data dumps from
sharepoint folder, else it fetches from postgres DB
threshold: it is used for selecting top Shap features, by default it has a value 0.7
nlargest: number of top Shap features
Returns:
A dataframe containing Shap explainability
"""
train_df_raw = pd.read_csv(train_data_path + "\Test_Data_02-11-2021-BASELINE-v0-6.csv")
train_df_raw = train_df_raw.loc[(train_df_raw["IsClosed"]==True) & (train_df_raw["CloseDate"]<"2021-11-02")]
logging.info("Explaining of OWR model started...")
# 1. Load and Preprocess data from different tables
try: 
test_df_preprocess = preprocessing_data(
train_set = False,
get_data_dumps = get_data_dumps,
data_dump_date = data_dump_date,
model_date = model_version_date
)
except Exception:
logging.error("Data loading and Preprocesing failed!", exc_info=True)
sys.exit(1)
#test_df_preprocess = pd.read_csv(test_data_path + "\c360_owr_preprocess.csv", sep = ";", decimal = ",")
# Remove Noises, Seed -> Close
if not open_opps_only:
test_df_raw = test_df_preprocess[test_df_preprocess["SeedStart-Close"] != True]
# Remove Seed Opps
test_df_raw = test_df_raw[test_df_preprocess["StageName"] != '0 - Seed']
# 2. Fill null values for test data
try:
test_df = fillna_data(test_df_raw, model_version_date, train_set=False)
except Exception:
logging.error("Imputation of missing values failed!", exc_info=True)
sys.exit(1)
# Load raw Salesforce output (only for the opportunity table)
opp_df = get_list_df("opportunity")
req_opp_df = opp_df[(opp_df["CRM_ID__c"].isna())]
# Get list of columns used in the model
categorical_cols, boolean_cols, numerical_cols = get_features(model_version_date)
req_model_cols = numerical_cols + boolean_cols + categorical_cols
# Missing value Imputation for training data
train_df = fillna_data(train_df_raw[req_model_cols], model_version_date, train_set = False)
# Load model
model_path = (
r"models/trained_models/"
+ model_version_date
+ r"/{application}.pkl".format(application=model)
)
prod_model= joblib.load(model_path)
# New category check
X_train = new_category_check(train_df, prod_model, model_version_date)
X_test = new_category_check(test_df, prod_model, model_version_date)
# Create a list of column names for categorical columns, after the one hot encoding
cat_dict={}
for col in categorical_cols:
cat_dict[col] = [col + "_" + x for x in train_df[col].unique()]
# Transform columms for test set
trf = prod_model["transformation"]
X_test_trf = trf.transform(X_test[req_model_cols])
# Get grouped Shap values for each data point
train_tree_shap_values_arr, grp_won_shap_df, grp_lost_shap_df = get_grp_shap(X_test_trf, X_train, X_test, cat_dict) 
# Returns the indices that would sort the Shap values
order_pos = np.argsort(-grp_won_shap_df.abs().values, axis=1)[:, :nlargest]
# Sort absolute Shap value by size
order_val = -np.sort(-grp_won_shap_df.abs().values, axis=1)[:, :nlargest]
order_val_thresh = -np.sort(-grp_won_shap_df.abs().values, axis=1)[:, :nlargest]
# Logic for selecting TOP features that meet the selected threshold together
for i in range(len(order_val)):
for n in range(nlargest):
if (order_val_thresh[i, :n+1].sum()/grp_won_shap_df.abs().values[i,:].sum()) >= threshold:
order_val_thresh[i, n+1:] = np.nan
order_val_thresh_df = pd.DataFrame(order_val_thresh,
columns = ['Top {} Feature or not'.format(i) for i in range(1, nlargest+1)])
col_list = order_val_thresh_df.columns
order_val_thresh_df.loc[:,"OpportunityId"] = test_df["OpportunityId"].values
order_val_thresh_long = pd.melt(order_val_thresh_df, id_vars = ["OpportunityId"], value_vars = col_list)
order_val_thresh_long = order_val_thresh_long.drop(columns=['OpportunityId', 'variable'])
# Get Feature names column
result_pos = pd.DataFrame(grp_won_shap_df.columns[order_pos],
columns=['Top {} Pos Feature Name'.format(i) for i in range(1, nlargest+1)],
index=grp_won_shap_df.index)
col_list = result_pos.columns
result_pos.loc[:,"OpportunityId"] = test_df["OpportunityId"].values

# Transform table from wide to long format
result_pos_long = pd.melt(result_pos, id_vars = ["OpportunityId"], value_vars = col_list)
result_pos_long.columns = result_pos_long.columns.str.replace('value', 'Feature name')
result_pos_long = result_pos_long.drop(columns=['variable'])

# Get Shap values
shap_value_list = []
for i in range(len(grp_won_shap_df)):
shap_value_list.append(grp_won_shap_df.iloc[i,order_pos[i]].values)
df_shaps = pd.DataFrame(shap_value_list, columns=['Top {} SHAP value'.format(i) for i in range(1, nlargest+1)])
col_list = df_shaps.columns
df_shaps.loc[:,"OpportunityId"] = test_df["OpportunityId"].values
df_shaps = df_shaps.set_index('OpportunityId', drop= False)
# Transform Shap table from wide to long format
df_shaps_long = pd.melt(df_shaps, id_vars = ["OpportunityId"], value_vars = col_list)
df_shaps_long.columns = df_shaps_long.columns.str.replace('value', 'Shap value')
df_shaps_long = df_shaps_long.drop(columns=['OpportunityId', 'variable'])
# Return original values for the features before any preprocessing
feature_value_list = []
for col in range(nlargest):
for i in range(len(result_pos)):
col_name = result_pos.iloc[i,col]
id = result_pos["OpportunityId"][i]
if col_name in req_model_cols:
# Show value as days and months if column equals to Opportunity Length (Months)
if col_name == "Opportunity Length (Months)":
og_value = str(test_df_raw[col_name].iloc[i].round(2))
elif col_name == "account_age_days":
og_value = test_df_raw[col_name].iloc[i]
elif col_name == "median_length_btw_orders_days":
og_value = test_df_raw[col_name].iloc[i]
elif col_name == "account_win_rate":
og_value = str(test_df_raw[col_name].iloc[i].round(2)*100) + '%'
# Get back original values from raw table before any categorical grouping
elif col_name == "Industry Subgroup":
og_value = req_opp_df["Industry_Subgroup__c"].loc[req_opp_df["Id"] == id].to_string(index=False)
elif col_name in numerical_cols:
og_value = test_df_raw[col_name].iloc[i].astype('int')
else:
og_value = test_df_raw[col_name].iloc[i]
feature_value_list.append(og_value) 
else:
og_col_name = result_pos_long["Feature name"][col].split(':')[0]
og_value = test_df_raw[col_name].iloc[i]
feature_value_list.append(og_value)
feature_value_list = pd.Series(feature_value_list)
# Combine all columns into one long table (Opp ID, Feature name, Feature value, Shap value, Shap threshold)
df_combined = pd.concat([result_pos_long, feature_value_list, df_shaps_long, order_val_thresh_long], axis=1, ignore_index=False)
df_combined.columns =['Opportunity ID', 'Feature name', 'Feature Value', 'SHAP value', 'SHAP threshold']
# Convert columns to string
df_combined = df_combined.astype({"Feature name": str}, errors='raise')
df_combined = df_combined.astype({"Feature Value": str}, errors='raise')
# Replace NA with North America in case of Segments (so isna method does not remove it)
df_combined["Feature Value"].replace("NA","North America")
# Remove features that we are not suppose to show
df_combined = df_combined.drop(df_combined[df_combined["Feature name"] == 'Opportunity Value (DKK)'].index)
df_combined = df_combined.drop(df_combined[df_combined["Feature name"] == 'Description exists'].index)
df_combined = df_combined.drop(df_combined[df_combined["Feature name"] == 'std_length_btw_orders_days'].index)
df_combined = df_combined.drop(df_combined[df_combined["Feature name"] == 'min_length_btw_orders_days'].index)
# Remove features related to Topics
df_combined = df_combined[~df_combined["Feature name"].str.contains("Topic")]
# Transform raw data table from wide to long
test_df_raw_long = test_df_raw[req_model_cols]
test_df_raw_long.loc[:,"OpportunityId"] = test_df_raw["OpportunityId"].values
test_df_raw_long = pd.melt(test_df_raw_long, id_vars = ["OpportunityId"], value_vars = req_model_cols)
# Create table for Missing values
missing_df = test_df_raw_long[test_df_raw_long["value"].isna()]
missing_df["SHAP value"] = 0
missing_df["SHAP threshold"] = 0
missing_df.columns =['Opportunity ID', 'Feature name', 'Feature Value', 'SHAP value', 'SHAP threshold']
missing_df['Feature Value'] = "missing"
# Drop rows related to features from missing df, which should not be shown
missing_df = missing_df.drop(missing_df[missing_df["Feature name"] == "Count of Status Change"].index)
missing_df = missing_df.drop(missing_df[missing_df["Feature name"] == "CallReport Median Interval (Days)"].index)
missing_df = missing_df.drop(missing_df[missing_df["Feature name"] == "Latest Connect (Days)"].index)
missing_df = missing_df.drop(missing_df[missing_df["Feature name"] == "Opportunity Length (Months)"].index)
# Drop all rows related to account characteristics
missing_df = missing_df.drop(missing_df[missing_df["Feature name"] == "account_win_rate"].index)
missing_df = missing_df.drop(missing_df[missing_df["Feature name"] == "count_shipped_orders"].index)
missing_df = missing_df.drop(missing_df[missing_df["Feature name"] == "average_shipped_orders"].index)
missing_df = missing_df.drop(missing_df[missing_df["Feature name"] == "account_age_days"].index)
missing_df = missing_df.drop(missing_df[missing_df["Feature name"] == "min_length_btw_orders_days"].index)
missing_df = missing_df.drop(missing_df[missing_df["Feature name"] == "median_length_btw_orders_days"].index)
missing_df = missing_df.drop(missing_df[missing_df["Feature name"] == "std_length_btw_orders_days"].index)
# Combine Shap and Missing tables
explain_output = df_combined.append(missing_df, ignore_index = True)
# Drop rows where values are NA/missing in the SHAP threshold and Opp Id columns
explain_output = explain_output.dropna(subset=['Opportunity ID', 'SHAP threshold'])
# Drop columns that are no longer needed
explain_output = explain_output.drop(columns=["SHAP threshold"])
# Replace values with aliases for feature names
feature_name_dict = {'CallReport Median Interval (Days)': 'Interval Between Call Reports',
'Count of CallReports': 'Call Report',
'Count of Status Change': 'Number of Status Changes',
'Description exists': 'Description',
'Forward Movement': 'Progression',
'Has Line Item': 'Product',
'Industry Subgroup': 'Industry Grouping',
'Latest Connect (Days)': 'Days since Last Connection',
'Opportunity Length (Months)': 'Opportunity Length',
'Topic Needs': 'Text about Needs',
'Topic Negotiation': 'Text about Negotiation',
'Topic Orders': 'Text about Orders',
'Topic Product': 'Text about Products',
'Topic Trial': 'Text about Trial',
'account_win_rate': 'Historical Account Win Rate',
'count_of_shipped_orders': 'Orders per Account',
'average_shipped_orders':'Yearly Orders per Account',
'account_age_days': 'Account Seniority',
'median_length_btw_orders_days': 'Length Between Orders',
}
explain_output["Feature name alias"] = explain_output["Feature name"]
explain_output = explain_output.replace({"Feature name alias": feature_name_dict})
# Replace values with aliases for feature values
feature_value_dict = {'False': 'not identified',
'Market Penetration (existing products)': 'Market Penetration',
'Market Share Gain (existing products)': 'Market Share Gain',
'New Product Launch (up to 3 years after launch date)': 'New Product Launch',
'True': 'identified',
'Unknown': 'missing'
}
explain_output["Feature Value alias"] = explain_output["Feature Value"]
explain_output = explain_output.replace({"Feature Value alias": feature_value_dict})
# Add zero for SHAP values of missing features
explain_output["SHAP value"].loc[explain_output["Feature Value alias"] == "missing"] = 0
# Format time interval columns
for i in range(len(explain_output)):
val = explain_output["Feature Value"].iloc[i]
if explain_output["Feature name"].iloc[i] == "Opportunity Length (Months)":
explain_output["Feature Value alias"].iloc[i] = format_time_to_months(float(val))
elif explain_output["Feature name"].iloc[i] == "account_age_days":
explain_output["Feature Value alias"].iloc[i] = format_time_to_years(float(val))
elif explain_output["Feature name"].iloc[i] == "median_length_btw_orders_days":
explain_output["Feature Value alias"].iloc[i] = format_time_to_years(float(val))
# Deal with missing values
explain_output[["Feature Value alias","Feature name alias"]] = explain_output[["Feature Value alias","Feature name alias"]].fillna("missing")
explain_output[["Feature Value alias","Feature name alias"]] = explain_output[["Feature Value alias","Feature name alias"]].replace("None","missing")
explain_output["Feature Value alias"] = explain_output["Feature Value alias"].replace("-2147483648","missing")
explain_output["Feature Value alias"] = explain_output["Feature Value alias"].replace("2147483648","missing")
# Add days string after numbers
explain_output["Feature Value alias"].loc[(explain_output["Feature name alias"] == "Days since Last Connection") & (explain_output["Feature Value alias"]!="missing")].apply(lambda row: row + " days" if abs(int(row)) > 1 else row + " day")
explain_output["Feature Value alias"].loc[(explain_output["Feature name alias"] == "Interval Between Call Reports") & (explain_output["Feature Value alias"]!="missing")].apply(lambda row: row + " days" if abs(int(row)) > 1 else row + " day")
# Change back North America to NA
explain_output.replace("North America", "NA")
# Sort rows by Opportunity ID
explain_output = explain_output.sort_values('Opportunity ID')
# Deal with missing and unknown duplicates
explain_output_final = explain_output.drop_duplicates(['Opportunity ID','Feature name alias','Feature Value alias'], keep = 'last')
print(explain_output_final["Feature Value"])
# Save output file - need to adjust to the Falcon folder
save_path = r"C:/Users/{user_name}/OneDrive - Novozymes A S/Desktop".format(user_name = user_name)
explain_output_final.to_csv(
save_path + r"/c360_owr_explainability.csv",
index=False,
sep=';',
decimal=',',
encoding='utf8'
)
model_version_date = "02-11-2021-BASELINE-v0-6"
data_dump_date = "02-11-2021"
get_data_dumps = False
explain_model(
model_version_date,
data_dump_date,
get_data_dumps,
threshold = 0.7,
nlargest = 6,
)
