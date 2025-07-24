import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def load_iris_dataset():
    possible_filenames = ['IRIS.csv', 'iris.csv']
    for fname in possible_filenames:
        try:
            df = pd.read_csv(fname)
            print(f" Loaded dataset using filename: '{fname}'. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"WARNING: Could not load '{fname}' due to format error: {e}")
    
    print("\nCould not find the file with common names. Listing directory contents to auto-detect...")
    try:
        for fname in os.listdir('.'):
            if "iris" in fname.lower() and fname.lower().endswith('.csv'):
                print(f"Attempting to load auto-detected candidate: '{fname}'...")
                try:
                    df = pd.read_csv(fname)
                    print(f"SUCCESS: Loaded dataset using auto-detected filename: '{fname}'. Shape: {df.shape}")
                    return df
                except Exception as e:
                    print(f"WARNING: Failed to load '{fname}' (possibly corrupted): {e}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not list directory contents: {e}")
    
    print("\nFATAL ERROR: The Iris CSV dataset could not be found or loaded.")
    return None

def prepare_data(df):
    df.columns = [col.lower().replace('.', '_').replace(' ', '_') for col in df.columns]
    expected_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'species'

    if not all(feature in df.columns for feature in expected_features):
        print(f"Error: Missing feature columns. Found columns: {df.columns.tolist()}")
        return None, None, None

    if target not in df.columns:
        print(f"Error: Target column '{target}' not found. Found columns: {df.columns.tolist()}")
        return None, None, None

    df.dropna(subset=expected_features + [target], inplace=True)
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    print(f"\n'Species' column encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X = df[expected_features]
    y = df[target]
    return X, y, le

def train_model(X_train, y_train):
    model = LogisticRegression(solver='liblinear', random_state=42, max_iter=200)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy on the test set: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    return y_pred

def predict_examples(model, le, reverse_map):
    example_data = pd.DataFrame([
        [5.1, 3.5, 1.4, 0.2],
        [6.3, 3.3, 6.0, 2.5],
        [5.5, 2.5, 4.0, 1.3]
    ], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    print("\nExample Iris Flowers to Predict:")
    print(example_data)

    preds = model.predict(example_data)
    probs = model.predict_proba(example_data)

    for i, row in example_data.iterrows():
        pred_label = preds[i]
        species_name = reverse_map[pred_label]
        confidence = probs[i][pred_label]
        print(f"\nExample Flower {i+1}:")
        print(f"  Measurements: Sepal L={row['sepal_length']}, Sepal W={row['sepal_width']}, Petal L={row['petal_length']}, Petal W={row['petal_width']}")
        print(f"  Predicted Species: {species_name}")
        print(f"  Confidence: {confidence * 100:.2f}%")

def show_predictions_detail(X_test, y_test, y_pred, reverse_map):
    test_df = pd.DataFrame(X_test).reset_index(drop=True)
    pred_species = [reverse_map[i] for i in y_pred]
    actual_species = [reverse_map[i] for i in y_test]

    result_df = test_df.copy()
    result_df['Actual_Species'] = actual_species
    result_df['Predicted_Species'] = pred_species

    print("\nFirst 10 Predictions from Test Set:")
    print(result_df.head(10).to_string())

    print("\nLast 10 Predictions from Test Set:")
    print(result_df.tail(10).to_string())

def classify_iris_flowers():
    df = load_iris_dataset()
    if df is None:
        return

    print("\n--- Initial Data Snapshot ---")
    print(df.head())
    print("\n--- Missing Values Check ---")
    print(df.isnull().sum())

    X, y, le = prepare_data(df)
    if X is None:
        return

    reverse_map = {v: k for k, v in zip(le.classes_, le.transform(le.classes_))}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"\nTraining/Test Set Split: X_train={X_train.shape}, X_test={X_test.shape}")

    model = train_model(X_train, y_train)
    print("Model training complete.")

    y_pred = evaluate_model(model, X_test, y_test, le)

    predict_examples(model, le, reverse_map)
    show_predictions_detail(X_test, y_test, y_pred, reverse_map)

    print("\n--- Classification Complete ---")

if __name__ == "__main__":
    classify_iris_flowers()
