import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve, auc
from sklearn.metrics import f1_score, roc_auc_score, r2_score

# Load the .npy file
#data = np.load("PrimeNet/results/86467_finetuned.npy") # Load PrimeNet (PhysioNet 2012)
data = np.load("src/PrimeNet/results/51115_finetuned.npy") # Load mTAN -> PrimeNet (PhysioNet 2012)

# Print the shape of the data
print("Shape of the data:", data.shape)

# Print a few elements from the beginning and end to inspect
print("First row (data[0]):", data[0])
print("Second row (data[1]):", data[1])

# Manually inspecting a few elements to determine the structure
first_row_unique_values = len(np.unique(data[0]))
second_row_unique_values = len(np.unique(data[1]))

print("Unique values in first row:", first_row_unique_values)
print("Unique values in second row:", second_row_unique_values)

y_pred = data[0]
y_true = data[1]

# Function to determine if the data is continuous or categorical
def is_continuous(y):
    return len(np.unique(y)) > 2  # This is a simple heuristic, you can adjust it based on your needs

# Check if y_true and y_pred are continuous or categorical
if is_continuous(y_true) or is_continuous(y_pred):
    # If the data is continuous, use regression metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    print(f"MSE: {mse:.10f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Correlation Coefficient: {correlation:.4f}")
else:
    # If the data is categorical, calculate classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auprc = auc(recall, precision)
    f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'weighted' for multi-class tasks
    roc_auc = roc_auc_score(y_true, y_pred, average='weighted')  # Use 'weighted' for multi-class tasks
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
