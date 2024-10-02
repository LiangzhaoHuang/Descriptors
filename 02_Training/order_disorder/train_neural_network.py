import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

ZT1 = 12
ZT2 = 6
ZT3 = 24
ZT4 = 12
ZT5 = 24
ZT6 = 8
ZT7 = 48
ZT8 = 6
ZT = ZT1 + ZT2 + ZT3

# Load your data
data_train_Ni = np.loadtxt('train_occu_data_Ni.txt')
data_test_Ni = np.loadtxt('test_occu_data_Ni.txt')
data_train_Cr = np.loadtxt('train_occu_data_Cr.txt')
data_test_Cr = np.loadtxt('test_occu_data_Cr.txt')
data_train = np.append(data_train_Ni, data_train_Cr, axis=0)
data_test = np.append(data_test_Cr, data_test_Cr, axis=0)

# Get the order based on the first index and shuffle
indices = np.arange(data_train.shape[0])
np.random.shuffle(indices)

# Split the data into input features and target variable
# X is the atom occupancy of nearest neighbors
# Y is the ordering information:  0 = disorder; > 0 = the variant of ordered structure
X_train = data_train[indices, 1:ZT+1] - 1 
X_test = data_test[:, 1:ZT+1] - 1
y_train = data_train[indices, 0]
y_test = data_test[:, 0]

for n in range(len(y_train)):
    if y_train[n] > 0: y_train[n] = 1

for n in range(len(y_test)):
    if y_test[n] > 0: y_test[n] = 1

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# Define the neural network architecture
model = Sequential()
model.add(Dense(20, input_dim=X_train.shape[1], activation='relu',
                kernel_regularizer=l2(0.0005)
                ))  # First hidden layer
model.add(Dropout(0.05))  # Add dropout with 5% dropout rate
#model.add(Dense(10, activation='relu'))  # Second hidden layer
model.add(Dense(1, activation='sigmoid',
                kernel_regularizer=l2(0.0005)
                ))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=300, validation_split=0.2)

# Save weights and biases for each layer
i_layer = 0
for i, layer in enumerate(model.layers):
    if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:  # Check if the layer has weights
        weights, biases = layer.get_weights()
        i_layer += 1
        np.savetxt(f'layer_{i_layer}_weights.txt', weights)
        np.savetxt(f'layer_{i_layer}_biases.txt', biases)

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Print classification report
print(classification_report(y_test, y_pred))

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
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Disorder', 'Order'], yticklabels=['Disorder', 'Order'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('Confusion_matrix.png', dpi=200)
plt.show()
