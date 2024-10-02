import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the data to scale input similarly
data = np.loadtxt('./train_occu_data.txt')
X = data[:, 1:] - 1  # Features: 1st and 2nd NNs

# Define and fit the scaler
#scaler = StandardScaler()
#scaler.fit(X)

# Test input (using the same input as data[3, 1:] from your example)
OCCU_NEIGH = data[1110, 1:] - 1  # Adjusting the same way as in training
#OCCU_NEIGH = scaler.transform([OCCU_NEIGH])[0]  # Scale the input
OCCU_NEIGH = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1])
OCCU_NEIGH = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
print("Actual label:", data[1110, 0])

# Load the weights and biases
WEIGHT_1 = np.loadtxt('./layer_1_weights.txt')
WEIGHT_2 = np.loadtxt('./layer_2_weights.txt')

print('WEIGHT_1 = ')
to_print = '{:10s}'.format('reshape((/')
for nl, line in enumerate(WEIGHT_1.transpose()):
    if nl == 0:
        to_print += ', '.join(['{:13.6E}'.format(s) for s in line]) + ', &\n'
    elif nl == len(WEIGHT_1.transpose()) - 1:
        to_print += '{:10s}'.format('') + ', '.join(['{:13.6E}'.format(s) for s in line]) + '  &\n'
    else:
        to_print += '{:10s}'.format('') + ', '.join(['{:13.6E}'.format(s) for s in line]) + ', &\n'
to_print += '{:10s}/), (/{:d},{:d}/))'.format('', WEIGHT_1.shape[0], WEIGHT_1.shape[1])
print(to_print)

print('WEIGHT_2 = ')
to_print = '{:10s}'.format('reshape((/')
for nl, line in enumerate(WEIGHT_2.transpose()):
    if nl == 0:
        to_print += ', '.join(['{:13.6E}'.format(s) for s in line]) + ', &\n'
    elif nl == len(WEIGHT_2.transpose()) - 1:
        to_print += '{:10s}'.format('') + ', '.join(['{:13.6E}'.format(s) for s in line]) + '  &\n'
    else:
        to_print += '{:10s}'.format('') + ', '.join(['{:13.6E}'.format(s) for s in line]) + ', &\n'
to_print += '{:10s}/), (/{:d},{:d}/))'.format('', WEIGHT_2.shape[0], WEIGHT_2.shape[1])
print(to_print)

BIAS_1 = np.loadtxt('./layer_1_biases.txt')
BIAS_2 = np.loadtxt('./layer_2_biases.txt')

print('BIAS_1 = ')
to_print = ', '.join(['{:13.6E}'.format(s) for s in BIAS_1])
print(to_print)

print('BIAS_2 = ')
to_print = ', '.join(['{:13.6E}'.format(s) for s in BIAS_2])
print(to_print)

# Layer 1 calculations
LAY_1_OUT = np.dot(OCCU_NEIGH, WEIGHT_1) + BIAS_1
LAY_1_OUT = np.maximum(LAY_1_OUT, 0)  # ReLU activation
print("LAY_1_OUT:", LAY_1_OUT)

# Layer 2 (output layer) calculations
LAY_2_OUT = np.dot(LAY_1_OUT, WEIGHT_2) + BIAS_2
softmax = np.exp(LAY_2_OUT) / np.sum(np.exp(LAY_2_OUT))

print("LAY_2_OUT:", LAY_2_OUT)
print("Sigmoid output (probability): {:6.3f}{:6.3f}{:6.3f}{:6.3f}{:6.3f}{:6.3f}{:6.3f}".format(*softmax))
print("Predicted class:", np.where(softmax == np.max(softmax))[0][0])
