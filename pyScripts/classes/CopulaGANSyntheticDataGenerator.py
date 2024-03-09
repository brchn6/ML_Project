
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
import pandas as pd 
import random
import os

#%%
class CopulaGANSyntheticDataGenerator:
    def __init__(self, train_data, label_column,minority_class_label, majority_class_label, boolean_columns=[], *args, **kwargs):
        self.min_class = train_data[train_data[label_column] == minority_class_label]
        self.maj_class = train_data[train_data[label_column] == majority_class_label]
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(train_data)
        for column in boolean_columns:
            self.metadata.update_column(column_name=column, sdtype='boolean')
        self.args = args
        self.kwargs = kwargs

    def fit_and_generate(self, epochs=100):
        ct_gan = CopulaGANSynthesizer(metadata=self.metadata, epochs=epochs, *self.args, **self.kwargs)
        ct_gan.fit(self.min_class)
        random_number = random.randint(100, 1000)
        synthetic_samples_cop = ct_gan.sample(len(self.maj_class) - (len(self.min_class) - random_number))
        return synthetic_samples_cop

    def evaluate_quality(self, synthetic_data):
        quality_report = evaluate_quality(real_data=self.min_class, synthetic_data=synthetic_data, metadata=self.metadata)
        return quality_report.get_score()

    def run_diagnostic(self, synthetic_data):
        diagnostic_report = run_diagnostic(real_data=self.min_class, synthetic_data=synthetic_data, metadata=self.metadata)
        return diagnostic_report.get_score()

    def generate_balanced_df(self, synthetic_data, train_data):
        balanced_train_set = pd.concat([train_data, synthetic_data])
        return balanced_train_set
    
    def export_balanced_df(self, balanced_train_set):
        name = 'balanced_train_set.csv'
        path = os.path.join(os.getcwd(), '..', 'data')
        balanced_train_set.to_csv(os.path.join(path, name), index=False)





#here im gona write the logical Q to ask if i want 
    