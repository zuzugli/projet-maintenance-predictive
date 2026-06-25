import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

df = pd.read_csv('data/raw/predictive_maintenance_v3.csv')

print("SHAPE:", df.shape)
print("\nDTYPES:\n", df.dtypes)
print("\nVALEURS MANQUANTES:\n", df.isnull().sum())
print("\nDOUBLONS:", df.duplicated().sum())
print("\nDESCRIBE:\n", df.describe().T)
print("\nFAILURE_WITHIN_24H distribution:")
print(df['failure_within_24h'].value_counts(normalize=True))