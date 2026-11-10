import hashlib, pandas as pd
files=["submissions/baseline_v0.csv","submissions/baseline_v0_hgb.csv","reports/m023/validation_oof.csv","data/processed/UPDATE-09/training_dataset.parquet","UPDATE-10/Test.csv","UPDATE-10/SampleSubmission.csv","UPDATE-10/Stations (1).csv","UPDATE-10/AdditionalHistory.zip","UPDATE-10/DataDictionary.txt","UPDATE-10/Mumbai_Air_Quality_Starter.ipynb"]
for f in files:
    print(hashlib.sha256(open(f,'rb').read()).hexdigest(), f)
for f in ["UPDATE-10/Test.csv","UPDATE-10/SampleSubmission.csv","UPDATE-10/Stations (1).csv"]:
    df=pd.read_csv(f)
    print(f, df.shape, list(df.columns))
    print(df.head(2).to_string())
df=pd.read_csv("UPDATE-10/Train.csv")
print("TRAIN", df.shape, list(df.columns))
print(df.timestamp_ist.min(), df.timestamp_ist.max())
print(df.pm25.describe().to_dict())
print("dups", df.duplicated().sum())
print("nulls", df.isna().sum().to_dict())
print("stations", df.location_id.unique().tolist())
print(open("UPDATE-10/DataDictionary.txt").read())
