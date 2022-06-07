# Importing necessary libraries used for data cleaning, and data visualization
import pandas as pd


# Importing library to split the data into training part and testing part.
from sklearn.model_selection import train_test_split

# Importing library to process the data (Normalize the data)
from sklearn.preprocessing import StandardScaler

# Importing Models (used for making prediction)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler

## Data Collection and Data Exploration
# Loading the data from CSV file to pandas dataframe
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
features = scaler.fit_transform(X)

# ## Splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
# "X_train" represents training data and "X_test" represents test data. "y_train" represents outcome (status like person have parkinsons disease or not) of training data while "y_test" represents outcome (status like person have parkinsons disease or not) of test data.

os =  RandomOverSampler(sampling_strategy=1)
X_train, y_train = os.fit_resample(X_train, y_train)
print(len(y_train[y_train==0]), len(y_train[y_train==1]))
print(len(X_train))


# # Model Building
# ## kNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# ## XGBoost Model
svm = SVC(gamma=0.4, C=100, kernel='rbf', probability=True)
svm.fit(X_train,y_train)

# Pickling the Model
import pickle
pickle.dump(svm, open('model_svm.pkl','wb'))
model_svm = pickle.load(open('model_svm.pkl','rb'))

pickle.dump(knn, open('model_knn.pkl','wb'))
model_knn = pickle.load(open('model_knn.pkl','rb'))
