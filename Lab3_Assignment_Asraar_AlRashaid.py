import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("LiverFibrosis.csv")
df.head(5)


print("Dataset shape:", df.shape)


print("Column names:")
print(df.columns.tolist())


df.info()


df.dtypes


df.describe(include="all")


df.isnull().sum()


target_column = "CLASS"
print(df[target_column].value_counts())


plt.figure(figsize=(7, 4))
df[target_column].value_counts().sort_index().plot(kind="bar")
plt.title("Distribution of Target Classes")
plt.xlabel("Class")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


numeric_columns = df.select_dtypes(include="number").columns
df[numeric_columns].hist(figsize=(12, 8))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


box_columns = [column for column in ["Age", "BMI", "ALT", "AST", "Albumin", "Platelets", "Bilirubin"] if column in df.columns]
plt.figure(figsize=(12, 6))
df[box_columns].boxplot()
plt.title("Boxplots of Numerical Features")
plt.ylabel("Values")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


X = df.drop(columns=[target_column])
y = df[target_column]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)


predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print()
print(classification_report(y_test, predictions))


cm = confusion_matrix(y_test, predictions)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
