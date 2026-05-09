import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# load dataset
df = pd.read_csv('student.csv')

# encode target
le = LabelEncoder()
df['dropout'] = le.fit_transform(df['dropout'])

# fitur dan label
X = df.drop('dropout', axis=1)
y = df['dropout']

# scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)