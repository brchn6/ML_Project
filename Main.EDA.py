#%%
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")

#path to data file
GETCWD = os.getcwd()
PathToData = os.path.join(GETCWD + "\\diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv")

#assing df
Maindf = pd.read_csv(PathToData)

sns.set_style("darkgrid")
plt.style.use("dark_background")


#%%
class SeeTheData:
    def __init__(self,df):
        self.df = df
        self.NumericCols = None
        self.ObjectsCols = None
    
    def Subsetting(self):
        #variable assigns (col name, col type ect.)
        ColumnName , Columndtype = self.df.columns , self.df.dtypes
        Columndtype = Columndtype.to_frame()

        #subsetting from main df to objects and numeric columns
        self.NumericCols = ColumnName[Columndtype[0] == "int64"].tolist()
        self.ObjectsCols = ColumnName[Columndtype[0] == "object"].tolist()

    def Display(self):
        print(f"The df.describe numeric colums look like this:{display(df.describe())}")
        print("------------------------------------------")
        print(f"The df.info look like this:{display(df.info())}")
        print("------------------------------------------")
        # print(f"The df.nunique look like this: {display(df.nunique())} ")
        print("------------------------------------------")


    def HistPlotOfNumericColumns(self):
        num_cols = len(self.NumericCols)
        cols_per_row = 1  # Number of columns per row
        num_rows = (num_cols + cols_per_row - 1) // cols_per_row  # Calculate the number of rows needed

        if num_cols > 0:
            # Create a subplot grid
            fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row * 5, num_rows * 5))
            axes = axes.flatten()  # Flatten the axes array for easy iteration

            # Plot each numeric column
            for i, col in enumerate(self.NumericCols):
                sns.histplot(data=self.df, x=col, ax=axes[i])
                axes[i].set_title(f"Title of the column: {col}")

            # Hide any unused axes
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.show()
        else:
            print("No numeric columns to plot.")

    def CountPlotOfObjectColumns(self):
        num_cols = len(self.ObjectsCols)
        cols_per_row = 1  # Number of columns per row
        num_rows = (num_cols + cols_per_row - 1) // cols_per_row  # Calculate the number of rows needed

        if num_cols > 0:
            # Create a subplot grid
            fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row * 5, num_rows * 5))
            axes = axes.flatten()  # Flatten the axes array for easy iteration

            # Plot each numeric column
            for i, col in enumerate(self.ObjectsCols):
                sns.countplot(data=self.df, x=col, ax=axes[i])
                axes[i].set_title(f"Title of the column: {col}")

            # Hide any unused axes
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.show()
        else:
            print("No objects columns to plot.")

a= SeeTheData(Maindf)
a.Subsetting()
# a.Display()
a.CountPlotOfObjectColumns()
# a.HistPlotOfNumericColumns()

#%%
df["discharge_disposition_id"][0:100]
#%%
# Simulate data from a bivariate Gaussian
x = dfDiag["diag_1"]
y = dfDiag["diag_3"]

# Draw a combo histogram and scatterplot with density contours
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=x, y=y, s=5, color=".15")
# sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
# sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)