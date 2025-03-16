def compute_accuracy_gap(history):
    train_acc = np.array(history.history['accuracy'])
    val_acc = np.array(history.history['val_accuracy'])
    
    accuracy_gaps = train_acc - val_acc
    return np.mean(accuracy_gaps)

# Comparar accuracy entre dos modelos
accuracy_gap_model_1 = compute_accuracy_gap(history1)
accuracy_gap_model_2 = compute_accuracy_gap(history2)

print(f"Accuracy Gap Modelo 1 (promedio): {accuracy_gap_model_1:.4f}")
print(f"Accuracy Gap Modelo 2 (promedio): {accuracy_gap_model_2:.4f}")
