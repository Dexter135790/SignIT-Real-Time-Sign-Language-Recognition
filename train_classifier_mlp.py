import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load data
data_dict = pickle.load(open('./data1.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Normalize data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_predict)
print('Accuracy: {}% of samples were classified correctly!'.format(accuracy * 100))

conf_matrix = confusion_matrix(y_test, y_predict)
print('\nConfusion Matrix:')
print(conf_matrix)

print('\nClassification Report:')
print(classification_report(y_test, y_predict))

# Save the model
with open('mlp_model1.p', 'wb') as f:
    pickle.dump({'model': model, 'confusion_matrix': conf_matrix}, f)