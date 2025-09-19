@echo off
source .\paper_env\Scripts\activate
cd /d D:\Projects\MoodMate\paper_code\model_training

python train_angry.py
python train_disgust.py
python train_fear.py
python train_happy.py
python train_neutral.py
python train_sad.py

echo All trainings complete!
pause
