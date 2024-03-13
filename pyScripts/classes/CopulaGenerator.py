
from sdv.single_table import CopulaGANSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.base import TransformerMixin
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
import pandas as pd 
import random
import os

 #%%

class CopulaGANSyntheticDataGenerator(TransformerMixin):
    def __init__(self, label_column, minority_class_label, majority_class_label, gans, boolean_columns=[], *args, **kwargs):
        self.label_column = label_column
        self.minority_class_label = minority_class_label
        self.majority_class_label = majority_class_label
        self.boolean_columns = boolean_columns
        self.gans = gans
        self.args = args
        self.kwargs = kwargs
        self.metadata = None
        self.min_class = None
        self.maj_class = None
        self.synthetic_samples = None
        self.balanced_train_set = None

    def fit(self, X, y=None):
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(X)
        for column in self.boolean_columns:
            self.metadata.update_column(column_name=column, sdtype='boolean')
        self.min_class = X[X[self.label_column] == self.minority_class_label]
        self.maj_class = X[X[self.label_column] == self.majority_class_label]
        return self

    def fit_and_generate(self, epochs=100):
        if self.gans == 'copula':
            ct_gan = CopulaGANSynthesizer(metadata=self.metadata, epochs=epochs)
        elif self.gans == 'ctgan':
            ct_gan = CTGANSynthesizer(metadata=self.metadata, epochs=epochs)
        else:
            raise ValueError("GANS type not supported. Choose between 'copula' and 'ctgan'")
        ct_gan.fit(self.min_class)
        random_number = random.randint(100, 1000)
        self.synthetic_samples = ct_gan.sample(len(self.maj_class) - (len(self.min_class) - random_number))
        return self.synthetic_samples

    def transform(self, X):
        if self.synthetic_samples is None:
            raise ValueError("transform method can't be called before fit_and_generate")
        
        self.balanced_train_set = pd.concat([X, self.synthetic_samples])
        return self.balanced_train_set

    def fit_transform(self, X, y=None, export=False, epochs=100, name=''):
        self.fit(X, y)
        self.fit_and_generate(epochs = epochs)
        transformed_data = self.transform(X)
        if export:
            name = name
            self.export_balanced_df(name)
        return transformed_data

    def export_balanced_df(self, name):
        name = (name + '.csv')
        path = os.path.join(os.getcwd(), '..', 'data')
        self.balanced_train_set.to_csv(os.path.join(path, name), index=False)   