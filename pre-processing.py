
import pandas as pd

df = pd.read_excel('./data/Online Retail.xlsx')

# Preprocess the dataset

df['Is_C_Present'] = (
    df['InvoiceNo']
    .astype(str)
    .apply(lambda x: 1 if x.find('C') != -1 else 0))

df_clean = (
    df
    # filter out non-positive quantity values
    .loc[df["Quantity"] > 0]
    # remove InvoiceNos starting with C
    .loc[df['Is_C_Present'] != 1]
    # column filtering
    .loc[:, ["InvoiceNo", "Description"]]
    # dropping all rows with at least one missing value
    .dropna()
)

df_final = pd.DataFrame
df_final = df_clean['Description']

df_final.to_csv('./data/processed-data.csv', index=False)
