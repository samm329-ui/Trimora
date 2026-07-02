import librosa
import numpy as np
from ..data.models import AudioQuality
from ..config import get_config


def measure_audio_quality(audio_path: str) -> AudioQuality:
    cfg = get_config().quality
    y, sr = librosa.load(audio_path, sr=get_config().audio.SAMPLE_RATE)
    duration = librosa.get_duration(y=y, sr=sr)

    if duration == 0:
        return AudioQuality(
            snr_db=cfg.DEFAULT_SNR_DB,
            speech_rate=cfg.DEFAULT_SPEECH_RATE,
            volume_rms=cfg.DEFAULT_VOLUME_RMS
        )

    volume_rms = float(np.mean(librosa.feature.rms(y=y)))

    S = np.abs(librosa.stft(y))
    D_harmonic, D_percussive = librosa.decompose.hpss(S)
    signal_power = np.sum(D_harmonic ** 2)
    noise_power = np.sum(D_percussive ** 2) + 1e-10
    snr_db = float(10 * np.log10(signal_power / noise_power))

    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    speech_rate = float(len(onsets) / duration)

    return AudioQuality(snr_db=snr_db, speech_rate=speech_rate, volume_rms=volume_rms)
