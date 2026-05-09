import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# load dataset
df = pd.read_csv('student_1000.csv')

# encode target
le = LabelEncoder()
df['dropout'] = le.fit_transform(df['dropout'])

# fitur dan label
X = df.drop('dropout', axis=1)
y = df['dropout']

# split data (WAJIB sebelum scaling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling (hanya dari data train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# KNN
# ======================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# ======================
# Random Forest
# ======================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ======================
# Evaluasi
# ======================
print('KNN Accuracy:', accuracy_score(y_test, y_pred_knn))
print('RF Accuracy:', accuracy_score(y_test, y_pred_rf))

print('\nKNN Report:\n', classification_report(y_test, y_pred_knn))
print('\nRF Report:\n', classification_report(y_test, y_pred_rf))