import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file = open('train_occu_data.txt').readlines()
data = np.zeros((len(file), 19))
for n, line in enumerate(file):
    line_data = line.split()
    for m in range(19):
        data[n, m] = float(line_data[m])

# Separate features and labels
X = data[:, 1:] - 1 # features: 12 1NN and 6 2NN occupancies
y = data[:, 0]   # labels: ordered information
for n in range(len(y)):
    if y[n] > 0: y[n] = 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred_test))

# Save the model weights and biases
coefficients = model.coef_
intercept = model.intercept_

np.savetxt('coefficients.txt', coefficients)
np.savetxt('intercept.txt', intercept)

# Save the entire model for future use
joblib.dump(model, 'logistic_regression_model.joblib')

# Plotting the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plotting training and testing accuracies
epochs = range(1, model.n_iter_[0] + 1)
plt.figure(figsize=(10, 7))
plt.plot(epochs, [train_accuracy] * len(epochs), 'bo-', label='Training Accuracy')
plt.plot(epochs, [test_accuracy] * len(epochs), 'ro-', label='Testing Accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()