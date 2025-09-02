import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

class Common_Data_Loading():
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train_class = None  # integer labels (0,1,2,...)
        self.y_test_class = None
        self.emotion_labels = None

    def load_and_transform(self):
        df=pd.read_csv(r"D:\Projects\MoodMate\paper_code\augmentation_code\final_extracted_features.csv")
        
        X=df.drop(columns=['label']).values
        y=df['label']
        
        mean_imputer=SimpleImputer(strategy='mean')
        X[np.isinf(X)]=np.nan
        X[:,84:85]=mean_imputer.fit_transform(X[:,84:85])
        
        scaling_model=joblib.load(r'D:\Projects\MoodMate\paper_code\models\scaler.pkl')
        label_encoding_model=joblib.load(r"D:\Projects\MoodMate\paper_code\models\label_encoder.pkl")
        
        X_scaled=scaling_model.transform(X)
        y_encoded=label_encoding_model.transform(y)
        self.emotion_labels = label_encoding_model.classes_

        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 5, 17))
        
        X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_encoded, test_size=0.2, random_state=42)

        self.X_train=X_train
        self.X_test=X_test

        self.y_train_class = np.argmax(y_train, axis=1)  # shape: (39600,)
        self.y_test_class = np.argmax(y_test, axis=1)    
