# Importing necessary libraries used for data cleaning, and data visualization
import pandas as pd

# Ignoring ununnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Importing library to split the data into training part and testing part.
from sklearn.model_selection import train_test_split

# Importing library to process the data (Normalize the data)
from sklearn.preprocessing import MinMaxScaler

# Importing Models (used for making prediction)
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


## Data Collection and Data Exploration
# Loading the data from CSV file to pandas dataframe
df = pd.read_csv("data.csv")

# # Seperating target column
X = df.drop(["name", "status"], axis=1)
y = df["status"]

# # Data Normalization
scaler = MinMaxScaler()
features = scaler.fit_transform(X)

# ## Splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
# "X_train" represents training data and "X_test" represents test data. "y_train" represents outcome (status like person have parkinsons disease or not) of training data while "y_test" represents outcome (status like person have parkinsons disease or not) of test data.



# # Model Building
# ## kNN
knn = KNeighborsClassifier(n_neighbors=8, metric='euclidean')
knn.fit(X_train, y_train)

# ## XGBoost Model
xgb = XGBClassifier()
xgb.fit(X_train,y_train)

# Pickling the Model
import pickle
pickle.dump(xgb, open('model_xgb.pkl','wb'))
model_xgb = pickle.load(open('model_xgb.pkl','rb'))

pickle.dump(knn, open('model_knn.pkl','wb'))
model_knn = pickle.load(open('model_knn.pkl','rb'))
