"""
This is a script that contains function tace 
we apply on the data in the main file
the first method removes unwanted columns and rows
the second method splits the data into train and test sets
"""
#%%
#---------------------------------Libraries--------------------------------
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# ---------------------------------Functions--------------------------------

class PreProcessingFunctions:
    def __init__(self):
        pass
    def remove_unwanted_columns_and_rows(self, df):
        Subset_df = df[df['diag_1'].str.contains('250') | df['diag_2'].str.contains('250') | df['diag_3'].str.contains('250')]
        df = Subset_df
        df.loc[df["readmitted"] == ">30" , "readmitted"] = "NO"
        df = df.reset_index(drop= True)
        df['readmitted'] = le.fit_transform(df[['readmitted']]) 
        return df

    def split_data(self, df, ColName):
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(df, df[ColName]):
            train_set = df.loc[train_index]
            test_set = df.loc[test_index]
        return train_set, test_set
    

