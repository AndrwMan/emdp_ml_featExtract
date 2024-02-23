import zipfile
import os
import librosa

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

def main():
    # Path to the ZIP file containing audio files
    zip_file_path = './data/cleaned.zip'

    # Directory to extract the contents of the ZIP file
    extract_directory = './data/extracted/'

    # Unzip the audio files
    unzip_audio_files(zip_file_path, extract_directory)
    print(extract_directory)

if __name__ == "__main__":
    main()
