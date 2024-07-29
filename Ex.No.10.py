# Step 1: Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a decision tree classifier on the training set
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Step 5: Make predictions on the testing set
y_pred = clf.predict(X_test)

# Step 6: Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
accuracy
