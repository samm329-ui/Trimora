import librosa
import numpy as np
from ..data.models import AudioFeatures, Segment


def compute_audio_features(segment: Segment, y: np.ndarray, sr: int) -> AudioFeatures:
    if segment.duration <= 0:
        return AudioFeatures()

    start_sample = max(0, int(segment.start * sr))
    end_sample = min(len(y), int(segment.end * sr))

    if start_sample >= end_sample:
        return AudioFeatures()

    seg_audio = y[start_sample:end_sample]

    onsets = librosa.onset.onset_detect(y=seg_audio, sr=sr, units='time')
    onset_count = len(onsets)
    speech_rate = onset_count / segment.duration

    rms = librosa.feature.rms(y=seg_audio)
    volume = float(np.mean(rms))

    global_rms = librosa.feature.rms(y=y)
    global_volume = float(np.mean(global_rms))
    volume_delta = volume - global_volume

    return AudioFeatures(
        speech_rate=speech_rate,
        volume=volume,
        volume_delta=volume_delta,
        onset_count=onset_count
    )


def compute_pause_after(segment: Segment, next_segment: Segment) -> float:
    if next_segment is None:
        return 0.0
    gap = next_segment.start - segment.end
    return max(0.0, gap)
