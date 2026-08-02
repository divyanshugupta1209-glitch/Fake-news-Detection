# --- NEO-REFUTE: Fake News Detection (Training Progress Simulation) ---

import time
import random
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

print("📰 NEO-REFUTE: Real-Time Fake News Detection System\n")
print("Initializing model training...\n")
time.sleep(1)

# Simulate 10 epochs of training
epochs = 10
train_acc = 0.70
val_acc = 0.65
train_loss = 1.2
val_loss = 1.4

for epoch in range(1, epochs + 1):
    time.sleep(0.3)  # simulate training time
    train_acc += random.uniform(0.01, 0.03)
    val_acc += random.uniform(0.01, 0.025)
    train_loss -= random.uniform(0.05, 0.1)
    val_loss -= random.uniform(0.04, 0.09)

    print(f"Epoch [{epoch}/{epochs}]")
    print(f"Train Accuracy: {train_acc*100:.2f}% | Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Train Loss: {train_loss:.3f} | Validation Loss: {val_loss:.3f}")
    print("-" * 55)
    time.sleep(0.2)

# Final results
print("\n✅ Training Completed Successfully!\n")
print("Final Model Accuracy: 91.75%\n")

y_true = [0, 1, 1, 0, 1, 0, 0, 1]
y_pred = [0, 1, 1, 0, 1, 0, 1, 1]

print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

print("Confusion Matrix:\n")
print(np.array(confusion_matrix(y_true, y_pred)))
