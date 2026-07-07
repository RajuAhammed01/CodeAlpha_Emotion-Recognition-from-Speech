"""
SPEECH EMOTION RECOGNITION SYSTEM
CNN-LSTM Hybrid Model for Audio Emotion Classification
Fixed to handle classes with insufficient samples
"""

import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense, 
                                     Dropout, BatchNormalization, Activation)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION BLOCK - ADJUST THESE PARAMETERS
DATASET_PATH = "files"  # Folder containing the audio files
MAX_PAD_LEN = 150       # Fixed time width for the network
N_MFCC = 40             # Number of MFCC features to extract
TEST_SIZE = 0.2         # Train/Test split ratio
BATCH_SIZE = 32         # Training batch size
EPOCHS = 80             # Maximum training epochs
RANDOM_STATE = 42       # For reproducibility
MIN_SAMPLES_PER_CLASS = 2  # Minimum samples required per class

# 1A. AUDIO AUGMENTATION HELPER
def augment_audio(audio, sample_rate):
    """Add data augmentation for better generalization"""
    # Add noise
    noise = np.random.randn(len(audio)) * 0.005
    audio_noise = audio + noise
    
    # Time stretching
    stretch_factor = np.random.uniform(0.8, 1.2)
    audio_stretch = librosa.effects.time_stretch(audio, rate=stretch_factor)
    
    # Pitch shifting
    pitch_shift = np.random.randint(-3, 3)
    audio_pitch = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift)
    
    return [audio_noise, audio_stretch, audio_pitch]

# 1. FEATURE EXTRACTION ENGINE
def extract_audio_features(file_path, max_pad_len=MAX_PAD_LEN, n_mfcc=N_MFCC):
    """
    Extracts MFCC features from audio file and ensures uniform shape.
    """
    try:
        # Load audio with optimal settings
        audio, sample_rate = librosa.load(file_path, sr=22050, duration=3.0)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Pad or truncate to ensure uniform shape
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        # Transpose to (Time Steps, Features)
        return mfccs.T 
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# 2. DATASET LOADER WITH CLEANING
def load_custom_dataset(base_directory="files"):
    """
    Loads audio files from the custom folder structure:
    files/[folder_name]/[emotion].wav
    
    Args:
        base_directory: Root directory containing the 'files' folder
    
    Returns:
        X: Feature array
        y: Label array
        file_info: Dictionary with file metadata
        class_counts: Dictionary with class distribution
    """
    features = []
    labels = []
    file_info = []
    
    print(f"Loading dataset from: {base_directory}")
    print("Extracting acoustic features... This may take a few minutes.\n")
    
    # Check if the directory exists
    if not os.path.exists(base_directory):
        print(f"ERROR: Directory '{base_directory}' not found!")
        print(f"   Current working directory: {os.getcwd()}")
        return np.array([]), np.array([]), [], {}
    
    # First pass: collect all files and their emotions
    all_files = []
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                emotion = file.replace('.wav', '')
                
                # Clean up emotion labels (remove duplicates like Joyfully(1))
                if '(1)' in emotion:
                    emotion = emotion.replace('(1)', '').strip()
                    print(f"   Renaming '{file}' to '{emotion}.wav'")
                
                all_files.append({
                    'path': file_path,
                    'emotion_raw': file.replace('.wav', ''),
                    'emotion_clean': emotion
                })
    
    # Get class distribution before filtering
    raw_emotions = [f['emotion_clean'] for f in all_files]
    class_counts = pd.Series(raw_emotions).value_counts().to_dict()
    
    print(f"\n   Initial class distribution:")
    for emotion, count in class_counts.items():
        print(f"   - {emotion}: {count} samples")
    
    # Filter out classes with insufficient samples
    valid_emotions = [emotion for emotion, count in class_counts.items() 
                     if count >= MIN_SAMPLES_PER_CLASS]
    
    # Remove classes with too few samples
    filtered_files = [f for f in all_files if f['emotion_clean'] in valid_emotions]
    
    if len(filtered_files) < len(all_files):
        removed = len(all_files) - len(filtered_files)
        print(f"\n   Removed {removed} files from classes with < {MIN_SAMPLES_PER_CLASS} samples")
        print(f"   Kept emotions: {valid_emotions}")
    
    # Process filtered files
    print("\n   Processing audio files...")
    for f in filtered_files:
        data = extract_audio_features(f['path'])
        if data is not None:
            features.append(data)
            labels.append(f['emotion_clean'])
            file_info.append({
                'file_path': f['path'],
                'folder': os.path.basename(os.path.dirname(f['path'])),
                'emotion_raw': f['emotion_raw'],
                'emotion_clean': f['emotion_clean']
            })
        
        # Progress indicator
        if len(features) % 50 == 0 and len(features) > 0:
            print(f"      Processed {len(features)} files...")
    
    # Final class distribution
    final_counts = pd.Series(labels).value_counts().to_dict()
    print(f"\n   Final class distribution after cleaning:")
    for emotion, count in final_counts.items():
        print(f"   - {emotion}: {count} samples")
    
    print(f"\n✓ Successfully processed {len(features)} audio files")
    print(f"✓ Emotions found: {set(labels)}")
    
    return np.array(features), np.array(labels), file_info, final_counts

# 3. MODEL ARCHITECTURE BUILDER
def build_model(input_shape, num_classes):
    """
    Builds the CNN-LSTM hybrid model architecture.
    """
    model = Sequential([
        # First CNN Block
        Conv1D(256, kernel_size=5, padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Second CNN Block
        Conv1D(128, kernel_size=5, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Third CNN Block
        Conv1D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # LSTM Layers
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model

# 4. TRAINING FUNCTION
def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Trains the model with callbacks for optimization.
    """
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    lr_reduction = ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=5,
        verbose=1,
        factor=0.5,
        min_lr=0.00001
    )
    
    print("\n" + "="*50)
    print("Starting Model Training...")
    print("="*50 + "\n")
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, lr_reduction],
        verbose=1
    )
    
    return history

# 5. VISUALIZATION AND EVALUATION
def plot_training_history(history):
    """Plots accuracy and loss curves from training history."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    
    # Accuracy plot
    ax[0].plot(history.history['accuracy'], label='Train Accuracy', 
               color='#107c41', linewidth=2.5)
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy', 
               color='#f58220', linewidth=2.5)
    ax[0].set_title('Model Learning Accuracy Curve', fontsize=14, fontweight='bold')
    ax[0].set_xlabel('Epochs', fontsize=12)
    ax[0].set_ylabel('Accuracy', fontsize=12)
    ax[0].legend(loc='lower right')
    ax[0].grid(True, linestyle='--', alpha=0.5)
    
    # Loss plot
    ax[1].plot(history.history['loss'], label='Train Loss', 
               color='#107c41', linewidth=2.5)
    ax[1].plot(history.history['val_loss'], label='Validation Loss', 
               color='#f58220', linewidth=2.5)
    ax[1].set_title('Categorical Loss Trajectory', fontsize=14, fontweight='bold')
    ax[1].set_xlabel('Epochs', fontsize=12)
    ax[1].set_ylabel('Loss', fontsize=12)
    ax[1].legend(loc='upper right')
    ax[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plots confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 11, "weight": "bold"},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Speech Emotion Classification Confusion Matrix', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Emotion', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Emotion', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return cm

def evaluate_model(model, X_test, y_test, label_encoder, file_info=None):
    """Comprehensive model evaluation with metrics and visualizations."""
    # Final evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print("\n" + "="*50)
    print(f" FINAL TEST ACCURACY: {test_acc*100:.2f}%")
    print("="*50)
    
    # Generate predictions
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, label_encoder.classes_)
    
    # Classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(
        y_true_classes, 
        y_pred_classes, 
        target_names=label_encoder.classes_,
        digits=3
    )
    print(report)
    
    return y_pred_classes, y_true_classes

def save_results_to_csv(y_pred, y_true, label_encoder, file_info=None, output_file='predictions.csv'):
    """Saves prediction results to a CSV file."""
    results = []
    if file_info:
        # Get test indices from file_info
        for i in range(len(y_pred)):
            if i < len(file_info):
                results.append({
                    'File': os.path.basename(file_info[i]['file_path']),
                    'Folder': file_info[i]['folder'],
                    'True_Emotion': label_encoder.inverse_transform([y_true[i]])[0],
                    'Predicted_Emotion': label_encoder.inverse_transform([y_pred[i]])[0],
                    'Correct': y_true[i] == y_pred[i]
                })
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    return df

# 6. PREDICTION FUNCTIONS
def predict_emotion(audio_path, model, label_encoder):
    """Predicts emotion for a single audio file."""
    features = extract_audio_features(audio_path)
    if features is None:
        return None, 0.0
    
    features = np.expand_dims(features, axis=0)
    predictions = model.predict(features, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    emotion = label_encoder.inverse_transform([predicted_class])[0]
    return emotion, confidence

def batch_predict(folder_path, model, label_encoder):
    """Predicts emotions for all audio files in a folder."""
    results = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                emotion, confidence = predict_emotion(file_path, model, label_encoder)
                results.append({
                    'File': file,
                    'Path': file_path,
                    'Predicted_Emotion': emotion,
                    'Confidence': confidence
                })
    
    df = pd.DataFrame(results)
    return df

# 7. LOAD SAVED MODEL FUNCTION
def load_saved_model(model_path='emotion_recognition_model.h5', 
                     encoder_path='label_encoder_classes.npy'):
    """Loads a previously saved model and label encoder."""
    from tensorflow.keras.models import load_model
    
    model = load_model(model_path)
    label_classes = np.load(encoder_path, allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_classes
    
    return model, label_encoder

# 8. MAIN EXECUTION PIPELINE
def main():
    """Main execution function - runs the complete pipeline."""
    print("\n" + "="*60)
    print("🎵 SPEECH EMOTION RECOGNITION SYSTEM")
    print("CNN-LSTM Hybrid Architecture")
    print("="*60)
    
    # Check if 'files' folder exists
    if not os.path.exists("files"):
        print(f"\n ERROR: 'files' folder not found in current directory!")
        print(f"   Current directory: {os.getcwd()}")
        return
    
    # Step 1: Load and process dataset
    print("\n[1] Loading Dataset...")
    X, y, file_info, class_counts = load_custom_dataset("files")
    
    # Check if data was loaded
    if len(X) == 0:
        print("\n ERROR: No audio files found!")
        return
    
    # Step 2: Preprocess labels
    print("\n[2] Preprocessing Labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    num_classes = len(label_encoder.classes_)
    
    print(f"   Emotion classes: {list(label_encoder.classes_)}")
    print(f"   Number of classes: {num_classes}")
    
    # Check if we have enough samples per class for stratification
    min_samples = min(class_counts.values())
    print(f"\n   Minimum samples in any class: {min_samples}")
    
    # Step 3: Train/Test split (with or without stratification)
    print("\n[3] Splitting Dataset...")
    
    # Use stratification only if all classes have at least 2 samples
    use_stratify = min_samples >= 2
    
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE,
            stratify=y_encoded
        )
        print("   Using stratified split")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE,
            stratify=None
        )
        print("   Using regular split (some classes may be unbalanced)")
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    
    # Step 4: Build model
    print("\n[4] Building Model Architecture...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, num_classes)
    model.summary()
    
    # Step 5: Train model
    print("\n[5] Training Model...")
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Step 6: Evaluate and visualize
    print("\n[6] Evaluating Model...")
    plot_training_history(history)
    y_pred, y_true = evaluate_model(model, X_test, y_test, label_encoder, file_info)
    
    # Step 7: Save model and results
    print("\n[7] Saving Model and Results...")
    try:
        model.save('emotion_recognition_model.h5')
        np.save('label_encoder_classes.npy', label_encoder.classes_)
        print("   ✓ Model saved as 'emotion_recognition_model.h5'")
        print("   ✓ Label encoder saved as 'label_encoder_classes.npy'")
        
        save_results_to_csv(y_pred, y_true, label_encoder, file_info)
    except Exception as e:
        print(f"   Could not save files: {e}")
    
    # Step 8: Example prediction
    print("\n[8] Model Ready for Predictions!")
    print("\n To use the model for prediction:")
    print("   emotion, confidence = predict_emotion('path/to/audio.wav', model, label_encoder)")
    
    # Test prediction on a sample file
    test_audio = None
    if file_info:
        for info in file_info[:5]:
            if os.path.exists(info['file_path']):
                test_audio = info['file_path']
                break
    
    if test_audio:
        print(f"\n   Testing on sample: {os.path.basename(test_audio)}")
        emotion, confidence = predict_emotion(test_audio, model, label_encoder)
        if emotion:
            print(f"   Predicted: {emotion} (Confidence: {confidence:.2%})")
    
    print("\n" + "="*60)
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)

# 9. ENTRY POINT (main part)
if __name__ == "__main__":
    # Check if 'files' folder exists
    if not os.path.exists("files"):
        print("\n" + "="*60)
        print(" WARNING: 'files' folder not found!")
        print("="*60)
        print("\nPlease create a folder named 'files' in the current directory.")
        print("Inside 'files', create subfolders with your audio files.")
        print("\nExpected structure:")
        print("   files/")
        print("   ├── folder1/")
        print("   │   ├── euphoric.wav")
        print("   │   ├── joyfully.wav")
        print("   │   └── sad.wav")
        print("   ├── folder2/")
        print("   │   └── surprised.wav")
        print("   └── ...")
        print("\nCurrent directory:", os.getcwd())
        print("="*60)
        
        answer = input("\nDo you want to try running anyway? (y/n): ")
        if answer.lower() == 'y':
            main()
    else:
        main()