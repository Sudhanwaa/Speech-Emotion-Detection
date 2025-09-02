from utils import Utilities
import joblib
import librosa
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")
utils_obj=Utilities()

# Dictionary holding all models for different emotions
models = {
    "angry": load_model(r"D:\Projects\MoodMate\paper_code\models\angry_model.h5"),
    "happy": load_model(r"D:\Projects\MoodMate\paper_code\models\happy_model.h5"),
    "sad": load_model(r"D:\Projects\MoodMate\paper_code\models\sad_model.h5"),
    "fear": load_model(r"D:\Projects\MoodMate\paper_code\models\fear_model.h5"),
    "neutral": load_model(r"D:\Projects\MoodMate\paper_code\models\neutral_model.h5"),
    "disgust": load_model(r"D:\Projects\MoodMate\paper_code\models\disgust_model.h5")

}
features = utils_obj.feature_extraction_and_concatenation("audio/neutral_6779.wav")
print("Feature Extraction completed !")
# Step 2: Scale features
scaled_features = utils_obj.scaling_values(features)
scaled_features = scaled_features.reshape(1, 5, 17)
print("Scaling completed !")
# Step 3: Pass to each emotion model
predictions = {}
for emotion, model in models.items():
    predictions[emotion] = model.predict(scaled_features)[0][0]
winner = max(predictions, key=predictions.get)
print(winner)