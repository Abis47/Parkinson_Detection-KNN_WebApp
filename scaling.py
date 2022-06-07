import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data.csv")

# # Seperating target column
X = df.drop(['PPE',
 'Shimmer:APQ5',
 'MDVP:PPQ',
 'Shimmer:DDA',
 'MDVP:Shimmer(dB)',
 'MDVP:APQ',
 'MDVP:RAP',
 'HNR',
 'MDVP:Jitter(Abs)',
 'Jitter:DDP',
 'Shimmer:APQ3',
 'name', 'status'], axis=1)
y = df["status"]

# # Data Normalization
scaler = StandardScaler()
scaler = scaler.fit(X)

import pickle
pickle.dump(scaler, open('scaling.pkl','wb'))
scaler = pickle.load(open('scaling.pkl','rb'))
