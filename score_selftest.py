# Self-test of the take ranking on synthetic audio — no GPU, runs in the image build and in CI.
# A backing in A minor at 96 BPM; a layer that plays its chords on its beat must outrank the same layer
# transposed a tritone, the same layer 120 ms late, and a near-silent take. Red here = the worker would
# deliver a worse take than one it had.
import numpy as np

import score

SR = 44100
BPM = 96
BEAT = 60 / BPM
PROG = [[57, 60, 64], [53, 57, 60], [50, 53, 57], [52, 56, 59]]    # Am F Dm E


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def tone(f, dur, decay=0.6):
    t = np.arange(int(dur * SR)) / SR
    return sum((0.5 ** k) * np.sin(2 * np.pi * f * (k + 1) * t) for k in range(4)) * np.exp(-t / decay)


def render(shift=0, delay=0.0, gain=1.0, bars=8, hits=True):
    y = np.zeros(int((bars * 4 * BEAT + 1) * SR))
    for b in range(bars):
        ch = PROG[b % 4]
        for k in range(4):
            at = int((b * 4 + k) * BEAT * SR + delay * SR)
            for m in ch:
                s = tone(hz(m + 12 + shift), BEAT * 0.9) * 0.2
                y[at:at + len(s)] += s[:len(y) - at]
            if hits:
                n = int(0.03 * SR)
                y[at:at + n] += np.random.default_rng(b * 4 + k).standard_normal(n) * 0.3 * np.exp(-np.arange(n) / 200)
    return (y * gain).astype(np.float32)


back = score.analyse_backing(render(), SR)
good = score.score_take(back, render(shift=0, hits=True), SR)
tritone = score.score_take(back, render(shift=6, hits=True), SR)
late = score.score_take(back, render(delay=0.12, hits=True), SR)
quiet = render(gain=1.0)
quiet[int(2 * SR):] *= 1e-4
silent = score.score_take(back, quiet, SR)

r = {k: score.rank_value('keyboard', v) for k, v in (('good', good), ('tritone', tritone), ('late', late), ('silent', silent))}
print({k: (v, s) for (k, v), s in zip(r.items(), (good, tritone, late, silent))})
assert good['harmony'] > tritone['harmony'] + 0.05, 'harmony must see the wrong key'
assert abs(good['lag_ms']) <= 10, 'a take on the beat has no lag'
assert r['good'] > r['tritone'] and r['good'] > r['late'] and r['good'] > r['silent'], 'ranking must prefer the take in key, on time, audible'
print('score selftest ok')
