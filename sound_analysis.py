import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

n_dim = 193
limit_percentage = 73.


# 사운드 특징 추출 (MFCC, Chromagram from a waveform, Mel-scaled power spectrogram, Spectral contrast, Tonnetz
def extract_feature(file):
    y, sample_rate = librosa.load(file)
    stft = np.abs(librosa.stft(y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sample_rate).T, axis=0)

    return mfccs, chroma, mel, contrast, tonnetz


# 사운드 분석
def parse_audio_files(file):
    mfccs, chroma, mel, contrast, tonnetz = extract_feature(file)
    ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    feature = ext_features

    return feature


def analyze(file):
    print("<분석 시작>")

    xhat = parse_audio_files(file)
    xhat = tf.reshape(xhat, [-1, 1, n_dim, 1])

    # 학습된 모델 불러오기
    model = load_model('sound_model.h5')

    # 신경망에 데이터 주입
    yhat = model.predict(xhat)

    sound_kind = ["None", "Car horn", "-", "Dog bark", "-",
                  "Engine idling", "Gun shot", "Jackhammer", "Siren"]
    # 결과
    percentage = np.round(yhat[0] * 100, 0)
    answer = dict(zip(sound_kind, percentage))

    print('파일 :', file)
    print('결과 :', answer)
    print("<분석 종료>")

    return answer

#    if yhat[0][np.argmax(yhat[0])] * 100 > limit_percentage:
#        return sound_kind[int(np.argmax(yhat[0]))]
#    else:
#        return "Unknown"
