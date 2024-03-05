import pandas as pd
from sdv.single_table import CTGANSynthesizer
import os

os.chdir('C:\ML_diabetes_Saar')
# Load your dataset
data = pd.read_csv('train_set_ctgan.csv')

from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)
metadata.validate_data(data=data)

from sdv.single_table import CTGANSynthesizer

synthesizer = CTGANSynthesizer(metadata)
synthesizer.fit(data)

from sdv.sampling import Condition

condition = Condition(
    num_rows=5,
    column_values={'readmitted': 'YES'}
)

synthetic_data = synthesizer.sample_from_conditions(
    conditions=[condition]
)
