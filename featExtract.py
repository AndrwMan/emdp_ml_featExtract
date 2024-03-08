import zipfile
import os
import librosa
import soundfile as sf
import numpy as np
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import mutual_info_classif

def unzip_audio_files(zip_path, extract_path):
    # Ensure the extraction directory exists
    os.makedirs(extract_path, exist_ok=True)

    # Extract the contents of the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def read_audio_file(file_path):
    # Read in the audio file using librosa
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    return audio_data, sample_rate

def trim_silence(audio_data):
    # Trim silence before and after the audio speech
    trimmed_data, _ = librosa.effects.trim(audio_data)
    return trimmed_data

def write_audio_file(file_path, audio_data, sample_rate):
    # Write the audio data to a new file
    sf.write(file_path, audio_data, sample_rate)

def compute_mfcc(audio_data, sample_rate, n_mfcc=13):
    # Compute MFCCs for the entire audio file
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    return mfccs

def normalize_mfcc(mfccs):
    # Apply Z-score normalization across rows (axis=1)
    mean = np.mean(mfccs, axis=1, keepdims=True)
    std = np.std(mfccs, axis=1, keepdims=True)
    normalized_mfccs = (mfccs - mean) / std

    return normalized_mfccs

def main():
    # Path to the ZIP file containing audio files
    zip_file_path = './data/cleaned.zip'

    # Directory to extract the contents of the ZIP file
    extract_directory = './data/extracted/'

    # Unzip the audio files
    unzip_audio_files(zip_file_path, extract_directory)
    print(extract_directory)

    # List all extracted audio files from the 'cleaned' subdirectory
    audio_files = [f for f in os.listdir(os.path.join(extract_directory, 'cleaned')) if f.endswith('.wav')]
    print(audio_files)

    # Directory to store trimmed audio files
    trimmed_directory = './data/extracted/trimmed/'
    os.makedirs(trimmed_directory, exist_ok=True)

    # Directory to store computed MFCCs
    mfcc_directory = './data/extracted/mfcc/'
    os.makedirs(mfcc_directory, exist_ok=True)

    # Directory to store computed normalized MFCCs
    normalized_mfcc_directory = './data/extracted/normalized_mfcc/'
    os.makedirs(normalized_mfcc_directory, exist_ok=True)

    # Initialize an empty list to store mean MFCC vectors
    mean_mfcc_vectors = []

    # Trim and write audio files
    for audio_file in audio_files:
        audio_file_path = os.path.join(extract_directory, 'cleaned', audio_file)
        audio_data, sample_rate = read_audio_file(audio_file_path)
        trimmed_audio_data = trim_silence(audio_data)

        # Write the trimmed audio to the new directory
        trimmed_file_path = os.path.join(trimmed_directory, audio_file.replace('.wav', '_enhanced.wav'))
        #write_audio_file(trimmed_file_path, trimmed_audio_data, sample_rate)

        # Compute MFCCs for the trimmed audio
        mfccs = compute_mfcc(trimmed_audio_data, sample_rate)

        # Write the computed MFCCs to the new directory
        mfcc_file_path = os.path.join(mfcc_directory, audio_file.replace('.wav', '_mfcc.npy'))
        #np.save(mfcc_file_path, mfccs)

        # Normalize the computed MFCCs
        normalized_mfccs = normalize_mfcc(mfccs)

        # Write the computed normalized MFCCs to the new directory
        normalized_mfcc_file_path = os.path.join(normalized_mfcc_directory, audio_file.replace('.wav', '_normalized_mfcc.npy'))
        #np.save(normalized_mfcc_file_path, normalized_mfccs)

        # Transpose to make features the columns
        transposed_mfccs = normalized_mfccs.T

        # Compute mean across rows to get mean MFCC vector
        mean_mfcc = np.mean(transposed_mfccs, axis=0)

        # Append the mean_mfcc vector to the list
        mean_mfcc_vectors.append(mean_mfcc)

    # Stack the mean MFCC vectors to create a data matrix
    data_matrix = np.vstack(mean_mfcc_vectors)
    print("Data Matrix:")
    print(data_matrix)
    print(f"Dimensions of Data Matrix: {data_matrix.shape}")

    # Use RandomForestClassifier for feature importance
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # When labels avaliable, use them for training
    # labels = ...  # labels here
    # X_train, _, y_train, _ = train_test_split(data_matrix, labels, test_size=0.2, random_state=42)
    # clf.fit(X_train, y_train)

    # No labels, attempt feature importance part
    clf.fit(data_matrix, np.zeros(data_matrix.shape[0]))  # Dummy labels as we're not using them for actual prediction

    # Get feature importance scores
    feature_importance_scores = clf.feature_importances_

    print(f"Feature Importance Scores:")
    print(feature_importance_scores)
    
    
	# Compute mutual information between each feature and a hypothetical class variable
    mi_scores = mutual_info_classif(data_matrix, np.zeros(data_matrix.shape[0]))

    # Sort features based on mutual information scores
    sorted_indices = np.argsort(mi_scores)[::-1]

    # Choose the top k features based on mutual information
    k_features = 5  
    selected_features = sorted_indices[:k_features]

    # Print or use the selected features as needed
    print(f"Selected Features based on Mutual Information:")
    print(selected_features)
    print("Mutual Information Scores:")
    print(mi_scores[selected_features])

if __name__ == "__main__":
    main()
