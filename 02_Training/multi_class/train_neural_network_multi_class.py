import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Load your data
data = pd.read_csv('train_occu_data.txt', delim_whitespace=True, header=None)

# Split the data into input features and target variable
# X is the atom occupancy of nearest neighbors
# Y is the ordering information:  0 = disorder; > 0 = the variant of ordered structure
X = data.iloc[:, 1:].values - 1
y = data.iloc[:, 0].values

# Standardize the features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Convert target variable to categorical
y = to_categorical(y, num_classes=7)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
#model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='softmax'))

# Compile the model with a proper loss function
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Save weights and biases for each layer
for i, layer in enumerate(model.layers):
    weights, biases = layer.get_weights()
    np.savetxt(f'layer_{i+1}_weights.txt', weights)
    np.savetxt(f'layer_{i+1}_biases.txt', biases)

# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test_labels)
print(f'Accuracy: {accuracy}')

# Print classification report
print(classification_report(y_test_labels, y_pred))

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Compute confusion matrix
cm = confusion_matrix(y_test_labels, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(7), yticklabels=range(7))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
