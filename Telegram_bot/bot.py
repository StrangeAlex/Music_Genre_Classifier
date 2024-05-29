import telebot
import os
from collections import Counter
import pandas as pd
from telebot.types import Audio

import numpy as np
import librosa
from pydub import AudioSegment
import pickle

from my_token import MY_TOKEN  # Replace it with your own to run the bot


def extract_audio_features(file_path):
    y, sr = librosa.load(file_path)

    rmse = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    features = {
        'rmse_mean': np.mean(rmse),
        'rmse_std': np.std(rmse),
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_centroid_std': np.std(spectral_centroid),
        'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
        'spectral_bandwidth_std': np.std(spectral_bandwidth),
        'rolloff_mean': np.mean(rolloff),
        'rolloff_std': np.std(rolloff),
        'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
        'zero_crossing_rate_std': np.std(zero_crossing_rate),
    }

    for i in range(1, 41):
        features[f'mfcc{i}_mean'] = np.mean(mfcc[i - 1])
        features[f'mfcc{i}_std'] = np.std(mfcc[i - 1])

    for i in range(1, 13):
        features[f'chroma{i}_mean'] = np.mean(chroma[i - 1])
        features[f'chroma{i}_std'] = np.std(chroma[i - 1])

    for i in range(1, 7):
        features[f'tonnetz{i}_mean'] = np.mean(tonnetz[i - 1])
        features[f'tonnetz{i}_std'] = np.std(tonnetz[i - 1])

    features['chroma_cqt_mean'] = np.mean(chroma_cqt)
    features['chroma_cqt_std'] = np.std(chroma_cqt)

    features['spectral_contrast_mean'] = np.mean(spectral_contrast)
    features['spectral_contrast_std'] = np.std(spectral_contrast)

    return features


def convert_to_wav(file_path):
    base, ext = os.path.splitext(file_path)
    if ext.lower() != '.wav':
        wav_path = base + '.wav'
        AudioSegment.from_file(file_path).export(wav_path, format="wav")
        return wav_path
    return file_path


def split_audio(file_path, clip_length=3000):
    audio = AudioSegment.from_wav(file_path)
    clips = []
    for i in range(0, len(audio), clip_length):
        clip = audio[i:i + clip_length]
        if len(clip) == clip_length:
            clip_path = f"{file_path[:-4]}_clip_{i // clip_length}.wav"
            clip.export(clip_path, format="wav")
            clips.append(clip_path)
    return clips


def predict_genre_distribution(file_path, model):
    file_path = convert_to_wav(file_path)
    clips = split_audio(file_path)
    genre_counter = Counter()

    for clip in clips:
        features = extract_audio_features(clip)
        sample = pd.DataFrame(columns=features)
        sample.loc[0] = features
        predicted_genre = model.predict(sample)[0]
        genre_counter[predicted_genre] += 1
        os.remove(clip)

    return genre_counter


with open('SVM_model_best.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

bot = telebot.TeleBot(MY_TOKEN)


def format_genre_counter(counter):
    result = "Here's what I think about this song:\n"
    data = list(sorted(counter.items(), key=lambda x: -x[1]))
    data_sum = sum([e[1] for e in data])
    data_normalized = list(map(lambda x: (x[0], round(100*x[1]/data_sum, 2)), data))
    for genre, count in data_normalized:
        result += f"{genre}: {count}%\n"
    return result


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Welcome! Send me an mp3 audio file and I'll predict the genre distribution.")


@bot.message_handler(content_types=['audio'])
def handle_audio(message: Audio):
    file_info = bot.get_file(message.audio.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    audio_file_path = f"{message.audio.file_id}.mp3"
    with open(audio_file_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    try:
        print(f"Started predicting {audio_file_path}!")
        genre_counter = predict_genre_distribution(audio_file_path, loaded_model)
        response = format_genre_counter(genre_counter)
    except Exception as e:
        response = f"An error occurred: {str(e)}"

    bot.reply_to(message, response)

    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)


bot.infinity_polling()
