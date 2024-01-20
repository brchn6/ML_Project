#%%
#Cleaning IDS_mapping variables:
#These columns have a lot many NA data if different values:
df['admission_type_id'] = df['admission_type_id'].replace([8,6],5)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace([18,26],25)
df['admission_source_id'] = df['admission_source_id'].replace([21,20,17,15],9)
