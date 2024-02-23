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

def trim_silence(audio_data):
    # Trim silence before and after the audio speech
    trimmed_data, _ = librosa.effects.trim(audio_data)
    return trimmed_data

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
	
	# Read and trim silence for the first audio file
    if audio_files:
        print("entered")
        first_audio_file_path = os.path.join(extract_directory, 'cleaned', audio_files[0])
        audio_data, sample_rate = read_audio_file(first_audio_file_path)

        # Trim silence
        trimmed_audio_data = trim_silence(audio_data)

if __name__ == "__main__":
    main()
