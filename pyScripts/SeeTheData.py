import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

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
