import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load Dataset
df = pd.read_csv('student_performance_dataset.csv')

# 2. Preprocessing
# Define explicit mappings for categorical columns to ensure consistency
mappings = {
    'Gender': {'Male': 0, 'Female': 1},
    'Parental_Education_Level': {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3},
    'Internet_Access_at_Home': {'No': 0, 'Yes': 1},
    'Extracurricular_Activities': {'No': 0, 'Yes': 1},
    'Pass_Fail': {'Fail': 0, 'Pass': 1}
}

# Apply mappings
for col, mapping in mappings.items():
    df[col] = df[col].map(mapping)

# 3. Feature Selection
# We exclude 'Student_ID' (useless for prediction) and 'Final_Exam_Score' (as it's unknown before the exam)
X = df[['Gender', 'Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores', 
        'Parental_Education_Level', 'Internet_Access_at_Home', 'Extracurricular_Activities']]
y = df['Pass_Fail']

# 4. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using Random Forest to capture complex patterns
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 5. Verify Performance
y_pred = model.predict(X_test)
print(f"Model Training Complete. Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# 6. Save Model and Metadata
model_data = {
    'model': model,
    'mappings': mappings,
    'features': list(X.columns)
}

with open('student_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("File 'student_model.pkl' created successfully.")