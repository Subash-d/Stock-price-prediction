import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv(r'stock_with_sentiments.csv')
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)


X = df[['Close', 'sentiment']]
y = df['Target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}


for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))



model_accuracies = {
    'Logistic Regression': 0.51,
    'Random Forest': 0.44,
    'SVM': 0.48,
    'KNN': 0.52,
    'Gradient Boosting': 0.45,
    'LSTM': 0.50 
}

# Plot
plt.figure(figsize=(10, 6))
models = list(model_accuracies.keys())
accuracies = list(model_accuracies.values())

bars = plt.bar(models, accuracies, color='skyblue')
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)


for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.01, f'{acc:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()
