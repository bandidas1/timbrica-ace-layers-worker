# How well a generated layer sits on the person's backing track — numbers the worker ranks takes by.
#
# ACE-Step lego decides by seed what it plays (upstream issue #1375: «1 good take in ~10 on a dense mix»), so a
# take is never trusted blind: every take gets these numbers and the worker keeps the best one per track.
#   beat     — normalised cross-correlation of onset envelopes (layer vs backing), best lag within ±60 ms;
#   harmony  — cosine of beat-synchronous chroma (layer vs backing) MINUS the mean over the backing's chroma
#              rolled ±1..±6 semitones: a layer in the right key/chords scores > 0, one that just «plays notes» ≈ 0;
#   presence — share of 1-s windows where the layer is audible (RMS within 30 dB of its loudest window);
#   lag_ms   — lag of the beat peak (the layer must not be late or early).
# Pure numpy + librosa: runs on the worker's CPU while the GPU makes the next take, and in the app repo's bench.
import numpy as np
import librosa

SR = 22050
HOP = 220          # 100 frames per second


def _mono(y, sr):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=0 if y.shape[0] <= 2 else 1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def _onset(y):
    return librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)


def _xcorr_peak(a, b, maxlag):
    n = min(len(a), len(b))
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0, 0
    best, bl = -1.0, 0
    for l in range(-maxlag, maxlag + 1):
        v = np.dot(a[l:], b[:n - l]) if l >= 0 else np.dot(a[:n + l], b[-l:])
        if v > best:
            best, bl = v, l
    return float(best / (na * nb)), bl


def analyse_backing(y, sr):
    """Everything about the backing the takes are compared with — computed once per job."""
    m = _mono(y, sr)
    on = _onset(m)
    tempo, beats = librosa.beat.beat_track(onset_envelope=on, sr=SR, hop_length=HOP)
    ch = librosa.feature.chroma_cqt(y=m, sr=SR, hop_length=HOP)
    return {'onset': on, 'beats': beats, 'chroma': ch, 'n': len(m)}


def _sync_chroma(ch, beats):
    if len(beats) < 4:
        return ch
    return librosa.util.sync(ch, beats, aggregate=np.mean)


def score_take(back, y, sr):
    m = _mono(y, sr)
    m = m[:back['n']] if len(m) > back['n'] else np.pad(m, (0, back['n'] - len(m)))
    beat, lag = _xcorr_peak(back['onset'], _onset(m), maxlag=6)
    ch = librosa.feature.chroma_cqt(y=m, sr=SR, hop_length=HOP)
    a = _sync_chroma(back['chroma'], back['beats'])
    b = _sync_chroma(ch, back['beats'])
    k = min(a.shape[1], b.shape[1])
    a, b = a[:, :k], b[:, :k]
    energy = b.sum(axis=0)
    keep = energy > (np.percentile(energy, 20) if k > 5 else 0)

    def cos(x, y_):
        num = (x * y_).sum(axis=0)
        den = np.linalg.norm(x, axis=0) * np.linalg.norm(y_, axis=0) + 1e-9
        return float(np.mean((num / den)[keep])) if keep.any() else 0.0

    h0 = cos(a, b)
    hs = [cos(np.roll(a, s, axis=0), b) for s in (1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5)]
    win = SR
    rms = np.array([np.sqrt(np.mean(m[i:i + win] ** 2) + 1e-12) for i in range(0, max(1, len(m) - win), win)])
    db = 20 * np.log10(rms + 1e-12)
    presence = float(np.mean(db > db.max() - 30)) if len(db) else 0.0
    return {'beat': round(beat, 3), 'lag_ms': lag * 10, 'harmony': round(h0 - float(np.mean(hs)), 3),
            'harmony_raw': round(h0, 3), 'presence': round(presence, 2),
            'rms_db': round(float(20 * np.log10(np.sqrt(np.mean(m ** 2)) + 1e-12)), 1)}


# Per-track weights: drums have no pitch, pads/strings have soft attacks.
WEIGHTS = {
    'drums': {'beat': 1.0, 'harmony': 0.0}, 'percussion': {'beat': 1.0, 'harmony': 0.0},
    'strings': {'beat': 0.3, 'harmony': 1.0}, 'keyboard': {'beat': 0.6, 'harmony': 1.0},
    'synth': {'beat': 0.5, 'harmony': 1.0}, 'guitar': {'beat': 0.8, 'harmony': 1.0},
    'bass': {'beat': 0.8, 'harmony': 1.0}, 'brass': {'beat': 0.5, 'harmony': 1.0},
    'woodwinds': {'beat': 0.4, 'harmony': 1.0},
}


def rank_value(track, s):
    w = WEIGHTS.get(track, {'beat': 0.5, 'harmony': 1.0})
    v = w['beat'] * s['beat'] + w['harmony'] * s['harmony'] * 4.0
    if s['presence'] < 0.5:
        v -= 1.0                       # mostly silent take
    if abs(s['lag_ms']) > 40:
        v -= 0.5                       # audibly early/late
    return round(v, 3)
