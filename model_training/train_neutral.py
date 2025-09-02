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
# Load data
loader = Common_Data_Loading()
loader.load_and_transform()

# Create binary labels for "neutral" (assuming neutral is class index 0)
target_emotion = "neutral"
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

# Example usage
model = create_strong_gru_model()

# Callbacks for better performance
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
checkpoint = ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True)


# Training example
history = model.fit(loader.X_train, y_train_bin, validation_data=(loader.X_test, y_test_bin),
                    epochs=50, batch_size=64, callbacks=[early_stop, checkpoint])

print(history)

# Confusion matrix & classification report
y_prob = model.predict(loader.X_test).ravel()       # probabilities
y_pred = (y_prob >= 0.5).astype(int)         # default threshold

print(classification_report(y_test_bin, y_pred))
print(confusion_matrix(y_test_bin, y_pred))


# Find best F1 threshold from precision–recall curve
prec, rec, thresholds = precision_recall_curve(y_test_bin, y_prob)
f1_scores = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]
print("best threshold", best_threshold, "best F1", f1_scores[best_idx])

# Plot PR curve
plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve")
plt.show()

# Saving File
model.save(r"D:\Projects\MoodMate\paper_code\models\neutral_model.h5")
print("Saved Model successfully")
