import numpy as np

y_true = eval(
    "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1]")
y_pred = eval(
    "[0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0]")

y_true, y_pred = np.array(y_true, dtype=int), np.array(y_pred, dtype=int)

accuracy = np.sum(y_true == y_pred) / len(y_true)

print(y_true[y_true == y_pred])
print("Accuracy:", accuracy)
positive_precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
positive_recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)

negative_precision = np.sum((y_true == -1) & (y_pred == -1)) / np.sum(y_pred == -1)
negative_recall = np.sum((y_true == -1) & (y_pred == -1)) / np.sum(y_true == -1)

print("Positive precision:", positive_precision)
print("Positive recall:", positive_recall)
print("Negative precision:", negative_precision)
print("Negative recall:", negative_recall)
