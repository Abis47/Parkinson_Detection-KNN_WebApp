import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data.csv")

# # Seperating target column
X = df.drop(["name", "status"], axis=1)
y = df["status"]

# # Data Normalization
scaler = MinMaxScaler()
scaler = scaler.fit(X)

import pickle
pickle.dump(scaler, open('scaling.pkl','wb'))
scaler = pickle.load(open('scaling.pkl','rb'))
