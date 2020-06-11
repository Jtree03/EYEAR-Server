import glob
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

from eyear_server import settings

n_dim = 193
limit_percentage = 73.
# 사운드 판단 모듈입니다.


# 사운드 특징 추출 (MFCC, Chromagram from a waveform, Mel-scaled power spectrogram, Spectral contrast, Tonnetz
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


# 사운드 분석
def parse_audio_files(filenames):
    rows = len(filenames)
    features = np.zeros((rows, 193))
    i = 0
    for fn in filenames:
        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        except:
            print("오류 발생")
            print(fn)
        else:
            features[i] = ext_features
            i += 1
            print("특징 추출 완료")
    return features


def main():
    print("<분석>")
    audio_files = []
    audio_files.extend(glob.glob(os.path.join(settings.BASE_DIR, '/EYEAR_server/media/file.wav')))

    X = parse_audio_files(audio_files)

    xhat = X
    xhat = tf.reshape(xhat, [-1, 1, n_dim, 1])

    # 학습된 모델 불러오기
    model = load_model(os.path.join(settings.BASE_DIR, '/EYEAR_server/learning/sound_model.h5'))

    # 신경망에 데이터 주입
    yhat = model.predict_proba(xhat)

    sound_kind = ["None", "Car horn", "-", "Dog bark", "-",
                  "Engine idling", "Gun shot", "Jackhammer", "Siren"]
    # 결과
    i = 0
    percentage = np.round(yhat[i] * 100, 0)
    print(yhat)
    for result in yhat:
        print(i, '파일 : ' + audio_files[i] + '\t\t결과 : ' + sound_kind[int(np.argmax(yhat[i]))])
        print("확률 : " + str(percentage))
        i += 1
    print("젤 높은거 : " + str(yhat[0][np.argmax(yhat[0])]))
    
    answer = dict(zip(sound_kind, percentage))
    
    print(answer)
    
    return answer
    
#    if yhat[0][np.argmax(yhat[0])] * 100 > limit_percentage:
#        return sound_kind[int(np.argmax(yhat[0]))]
#    else:
#        return "Unknown"
