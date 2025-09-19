from common_data_loading import Common_Data_Loading
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Bidirectional,SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, roc_auc_score
)
import seaborn as sns
import pandas as pd 
# Load data
loader = Common_Data_Loading()
loader.load_and_transform()

# Create binary labels for "sad" (assuming sad is class index 0)
target_emotion = "sad"
emotion_idx = np.where(loader.emotion_labels == target_emotion)[0][0]

y_train_bin = (loader.y_train_class == emotion_idx).astype(int)
y_test_bin = (loader.y_test_class == emotion_idx).astype(int)

# Build model
def create_strong_gru_model(input_shape=(5, 17)):
    model = Sequential()

    # 1st GRU Layer (Bidirectional for better context)
    model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # 2nd GRU Layer
    model.add(Bidirectional(GRU(64, return_sequences=False)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Fully connected layers
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Output layer for binary classification
    model.add(Dense(1, activation="sigmoid"))

    # Compile with a tuned learning rate
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "Precision", "Recall"])

    return model
# ----------------------------
# Model Training
# ----------------------------
model = create_strong_gru_model()

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True)

history = model.fit(
    loader.X_train, y_train_bin,
    validation_data=(loader.X_test, y_test_bin),
    epochs=100, batch_size=32,
    callbacks=[early_stop, rlr, checkpoint]
)

# ----------------------------
# Predictions
# ----------------------------
y_prob = model.predict(loader.X_test).ravel()       # predicted probabilities
y_pred = (y_prob >= 0.5).astype(int)               # default threshold

# ----------------------------
# Classification Report
# ----------------------------
print("Classification Report:\n", classification_report(y_test_bin, y_pred))

# Convert classification report into DataFrame for LaTeX/table export
report_dict = classification_report(y_test_bin, y_pred)
with open(r"D:\Projects\MoodMate\paper_code\Plots\sad\classification_report.txt", "w") as f:
    f.write("Classification Report\n")
    f.write(report_dict)

# ----------------------------
# Confusion Matrix (Heatmap)
# ----------------------------
cm = confusion_matrix(y_test_bin, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(r"D:\Projects\MoodMate\paper_code\Plots\sad\confusion_matrix.pdf")
plt.close()

# ----------------------------
# Training Curves
# ----------------------------
plt.figure(figsize=(8, 4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs"); plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.savefig(r"D:\Projects\MoodMate\paper_code\Plots\sad\training_vs_validation_accuracy_curve.pdf")

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs"); plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig(r"D:\Projects\MoodMate\paper_code\Plots\sad\training_vs_validation_loss_curve.pdf")


prec, rec, thresholds = precision_recall_curve(y_test_bin, y_prob)
f1_scores = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]

plt.figure(figsize=(6,5))
plt.plot(rec, prec, label="PR Curve")
plt.scatter(rec[best_idx], prec[best_idx], marker="o", color="red",
            label=f"Best F1={f1_scores[best_idx]:.3f} at Th={best_threshold:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend()
plt.tight_layout()
plt.savefig(r"D:\Projects\MoodMate\paper_code\Plots\sad\precision_recall_curve.pdf")
plt.close()

print(f"Best threshold = {best_threshold:.2f}, Best F1 = {f1_scores[best_idx]:.3f}")


# ----------------------------
# ROC Curve
# ----------------------------
fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
auc_score = roc_auc_score(y_test_bin, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend()
plt.tight_layout()
plt.savefig(r"D:\Projects\MoodMate\paper_code\Plots\sad\ROC_curve.pdf")
plt.close()

print(f"ROC AUC Score = {auc_score:.3f}")
# ----------------------------
# Save Model
# ----------------------------
model.save(r"D:\Projects\MoodMate\paper_code\models\sad_model.h5")
print("Saved Model successfully")
