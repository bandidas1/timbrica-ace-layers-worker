# RunPod serverless handler: «Улучшить минус» — new instrument layers that play along with a person's backing track.
#
# Engine: ACE-Step 1.5 (MIT, code and weights), task `lego` («Generate the {TRACK} track based on the audio
# context»), base checkpoint. The worker never returns the backing itself: it returns separate layers, and the page
# mixes them over the UNTOUCHED original (the person's file stays bit-exact under the layers).
#
# lego decides by seed what it plays (upstream issue #1375), so each track is generated as N takes and every take is
# scored against the backing (score.py: beat, harmony, presence); the best one per track is delivered.
#
# Input
#   {"probe": true}                                       -> at once (RunPodApi::probeServes)
#   {"source_url" | "source_b64",                         signed GET of the backing (any ffmpeg-readable audio) | base64
#    "tracks": [{"name": "strings", "caption": "..."}],   names from TRACKS below
#    "takes": 3, "steps": 40, "guidance": 7.0, "shift": 1.0, "use_adg": false, "seed_base": 1000,
#    "model": "base" | "xl", "bpm": null, "keyscale": "",
#    "result_urls": {"strings": signed PUT, ...},         production: the best take per track, FLAC (StemTransfer)
#    "mix_db": -3, "min_dce": 0.0, "min_dpq": -0.15,      a take is delivered only if backing+layer (layer at mix_db
#                                                         RMS under the backing) scores ≥ the backing on CE/PQ
#                                                         AND PANNs hears the asked instrument, not a voice
#    "return_audio": "best" | "all" | "none",             bench: takes inline as base64 Opus
#    "deadline_at": unix seconds}
# Output {"ok": true, "audio_s", "gpu_s", "gpu", "model", "improved", "aes_backing",
#         "tracks": {name: {"best": i, "improved", "bytes", "takes": [score...]}}}  or  {"error": code}
#        A track whose best take did not improve the mix is NOT uploaded (improved: false).
import base64
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import librosa
import soundfile as sf
import torch

import layers_io
import score

ACE_ROOT = os.environ.get('ACE_ROOT', '/app/ace')
sys.path.insert(0, ACE_ROOT)
os.chdir(ACE_ROOT)
from acestep.handler import AceStepHandler                                   # noqa: E402
from acestep.inference import GenerationParams, GenerationConfig, generate_music  # noqa: E402

TRACKS = ('strings', 'keyboard', 'drums', 'bass', 'guitar', 'synth', 'brass', 'woodwinds', 'percussion')
MODELS = {'base': 'acestep-v15-base', 'xl': 'acestep-v15-xl-base'}
MAX_SECONDS = float(os.environ.get('LAYERS_MAX_SECONDS', '480'))
MAX_TAKES = int(os.environ.get('LAYERS_MAX_TAKES', '6'))
GPU = torch.cuda.get_device_name(0)

# Meta's audiobox-aesthetics (CC BY 4.0, facebook/audiobox-aesthetics, baked in the image): the take is judged by
# what the MIX does — backing + layer against the backing alone (CE content enjoyment, PQ production quality).
# Bench 19.11.2026: on cheap / hollow backings good takes add CE +0.7…+2.2; on an already full production every
# take made it worse (CE −0.2…−1.4) — a layer that makes the person's music worse is not delivered.
from audiobox_aesthetics.infer import initialize_predictor  # noqa: E402
AES = initialize_predictor()


def _aes(arr, sr):
    r = AES.forward([{'path': torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)), 'sample_rate': sr}])[0]
    return {k: float(v) for k, v in r.items()}


def _mix(back, layer, db):
    n = min(back.shape[1], layer.shape[1])
    b, l = back[:, :n], layer[:, :n]
    if l.shape[0] == 1:
        l = np.vstack([l, l])
    rb = float(np.sqrt(np.mean(b ** 2)) + 1e-12)
    rl = float(np.sqrt(np.mean(l ** 2)) + 1e-12)
    m = b + l * (rb / rl) * (10 ** (db / 20))
    peak = float(np.max(np.abs(m))) or 1.0
    return m / max(1.0, peak / 0.95)

# PANNs Cnn14 (code MIT, weights CC BY 4.0, Kong et al. 2020; baked in the image): is the take the instrument the
# person asked for, and does it not sing. Bench 20.11.2026: chosen by the mix alone, 25–75 % of the delivered
# «strings» / «drums» were heard by Cnn14 as something else (a synth pad, a piano, a choir); with this gate the
# share of tracks delivered moved by a few points and every delivered take was the asked family.
from panns_inference import AudioTagging, labels as PANN_LABELS  # noqa: E402
PANNS = AudioTagging(checkpoint_path=os.path.join(os.path.expanduser('~'), 'panns_data', 'Cnn14_mAP=0.431.pth'), device='cuda')
FAMILY = {
    'strings': ['Violin, fiddle', 'Bowed string instrument', 'String section', 'Cello', 'Orchestra', 'Double bass'],
    'keyboard': ['Piano', 'Electric piano', 'Keyboard (musical)', 'Synthesizer', 'Organ', 'Electronic organ', 'Hammond organ'],
    'drums': ['Drum kit', 'Drum machine', 'Drum', 'Snare drum', 'Bass drum', 'Hi-hat', 'Cymbal', 'Percussion', 'Drum roll'],
    'bass': ['Bass guitar', 'Double bass'],
    'guitar': ['Guitar', 'Electric guitar', 'Acoustic guitar', 'Plucked string instrument'],
    'synth': ['Synthesizer', 'Electronic music'],
    'brass': ['Brass instrument', 'Trumpet', 'Trombone', 'French horn'],
}
SINGING = ['Singing', 'Male singing', 'Female singing', 'Vocal music', 'Speech', 'Choir', 'Synthetic singing']
FAM_MIN, SING_MAX = 0.15, 0.2


def _instrument(layer, sr, track):
    """{fam: share of the asked family in Cnn14's top-10 mass, sing: the strongest voice label, top: 3 labels}."""
    m = layer.mean(0)
    if sr != 32000:
        m = librosa.resample(m, orig_sr=sr, target_sr=32000)
    clip, _ = PANNS.inference(m[None, :].astype(np.float32))
    pr = clip[0]
    top = np.argsort(pr)[::-1][:10]
    mass = float(pr[top].sum()) + 1e-9
    fam = sum(float(pr[PANN_LABELS.index(l)]) for l in FAMILY.get(track, []) if PANN_LABELS.index(l) in top) / mass
    sing = max(float(pr[PANN_LABELS.index(l)]) for l in SINGING)
    return {'fam': round(fam, 3), 'sing': round(sing, 3), 'top': [PANN_LABELS[i] for i in top[:3]]}


_H = {'h': None, 'model': None}


def _ensure_model(key):
    if _H['model'] == key:
        return _H['h']
    cfg = MODELS[key]
    ckpt = os.path.join(ACE_ROOT, 'checkpoints', cfg)
    if not os.path.isdir(ckpt):
        # Only the bench asks for a checkpoint that is not baked in; fetch it once per worker.
        from huggingface_hub import snapshot_download
        snapshot_download('ACE-Step/' + cfg, local_dir=ckpt)
    h = AceStepHandler()
    msg, ok = h.initialize_service(project_root=ACE_ROOT, config_path=cfg, device='cuda',
                                   offload_to_cpu=False, offload_dit_to_cpu=False)
    if not ok:
        raise RuntimeError('model_init_failed: ' + str(msg)[:200])
    _H.update(h=h, model=key)
    return h


def _decode(path, td):
    """Any audio -> 48 kHz stereo WAV (what ACE reads), via ffmpeg; returns (wav_path, seconds)."""
    out = os.path.join(td, 'backing.wav')
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-vn', '-ac', '2', '-ar', '48000', out],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        raise ValueError((r.stderr or 'decode failed')[-200:])
    info = sf.info(out)
    return out, info.frames / info.samplerate


def _encode(wav_path, fmt, td, name):
    if fmt == 'flac':
        out = os.path.join(td, name + '.flac')
        args = ['-c:a', 'flac', '-sample_fmt', 's16']
    elif fmt == 'opus':
        out = os.path.join(td, name + '.ogg')
        args = ['-c:a', 'libopus', '-b:a', '96k']
    else:
        out = os.path.join(td, name + '.m4a')
        args = ['-c:a', 'aac', '-b:a', '256k', '-movflags', '+faststart']
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', wav_path] + args + [out], check=True)
    return out


def _take(h, src, track, caption, seed, p, td):
    params = GenerationParams(
        task_type='lego', src_audio=src,
        instruction='Generate the %s track based on the audio context:' % track.upper(),
        caption=caption, lyrics='[Instrumental]', instrumental=True,
        repainting_start=0.0, repainting_end=-1,
        inference_steps=int(p['steps']), guidance_scale=float(p['guidance']), shift=float(p['shift']),
        use_adg=bool(p['use_adg']), bpm=p.get('bpm'), keyscale=p.get('keyscale') or '',
        thinking=False, use_cot_metas=False, use_cot_caption=False, use_cot_language=False,
    )
    cfg = GenerationConfig(batch_size=1, audio_format='wav', use_random_seed=False, seeds=[seed])
    r = generate_music(h, None, params, cfg, save_dir=os.path.join(td, 'gen_%s_%d' % (track, seed)))
    if not r.success or not r.audios:
        raise RuntimeError('generation_failed: ' + str(r.error)[:200])
    return r.audios[0]['path']


def handler(job):
    inp = job.get('input') or {}
    if inp.get('probe'):
        a = torch.ones(64, 64, device='cuda')
        return {'ok': True, 'probe': True, 'gpu': GPU, 'sum': float((a @ a).sum())}
    tracks = [t for t in (inp.get('tracks') or []) if t.get('name') in TRACKS]
    if not tracks:
        return {'error': 'no_tracks'}
    if not inp.get('source_url') and not inp.get('source_b64'):
        return {'error': 'missing_source'}
    deadline = float(inp.get('deadline_at') or 0)
    p = {'steps': inp.get('steps', 40), 'guidance': inp.get('guidance', 7.0), 'shift': inp.get('shift', 1.0),
         'use_adg': inp.get('use_adg', False), 'bpm': inp.get('bpm'), 'keyscale': inp.get('keyscale')}
    takes = max(1, min(MAX_TAKES, int(inp.get('takes', 3))))
    seed_base = int(inp.get('seed_base', 1000))
    model = inp.get('model', 'base') if inp.get('model', 'base') in MODELS else 'base'
    want = inp.get('return_audio', 'none')
    mix_db = float(inp.get('mix_db', -3.0))
    min_dce = float(inp.get('min_dce', 0.0))
    min_dpq = float(inp.get('min_dpq', -0.15))
    results = inp.get('result_urls') or {}
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, 'source')
            if inp.get('source_url'):
                layers_io.download(inp['source_url'], raw, deadline=deadline)
            else:
                open(raw, 'wb').write(base64.b64decode(inp['source_b64']))
            try:
                src, seconds = _decode(raw, td)
            except ValueError as e:
                return {'error': 'source_unreadable', 'detail': str(e)[:160]}
            if seconds > MAX_SECONDS:
                return {'error': 'too_long', 'seconds': round(seconds)}
            y, sr = sf.read(src, always_2d=True, dtype='float32')
            back = score.analyse_backing(y.T, sr)
            aes0 = _aes(y.T, sr)
            h = _ensure_model(model)
            out = {}
            gpu_s = 0.0
            for t in tracks:
                name = t['name']
                caption = (t.get('caption') or name)[:400]
                rows = []
                for k in range(takes):
                    if deadline and time.time() > deadline:
                        return {'error': 'deadline_passed'}
                    g0 = time.time()
                    path = _take(h, src, name, caption, seed_base + k, p, td)
                    torch.cuda.synchronize()
                    gpu_s += time.time() - g0
                    ty, tsr = sf.read(path, always_2d=True, dtype='float32')
                    s = score.score_take(back, ty.T, tsr)
                    s['rank'] = score.rank_value(name, s)
                    if tsr == sr:
                        a = _aes(_mix(y.T, ty.T, mix_db), sr)
                        s['dCE'] = round(a['CE'] - aes0['CE'], 3)
                        s['dPQ'] = round(a['PQ'] - aes0['PQ'], 3)
                        s.update(_instrument(ty.T, tsr, name))
                        right = name not in FAMILY or (s['fam'] >= FAM_MIN and s['sing'] < SING_MAX)
                        s['ok'] = bool(s['dCE'] >= min_dce and s['dPQ'] >= min_dpq and right)
                        # the mix decides; the fit numbers break ties and keep a lucky noise out
                        s['rank'] = round(s['dCE'] + s['dPQ'] + 0.25 * s['rank'], 3)
                    s['seed'] = seed_base + k
                    s['_path'] = path
                    rows.append(s)
                # the best of the takes that passed — a better-ranked take of the wrong instrument does not
                # hide a good one (v2 picked the argmax and dropped the track when it failed)
                passed = [i for i, r in enumerate(rows) if r.get('ok', True)]
                pool = passed or list(range(len(rows)))
                best = max(pool, key=lambda i: rows[i]['rank'])
                improved = bool(passed)
                entry = {'best': best, 'improved': improved, 'takes': [{k: v for k, v in r.items() if k != '_path'} for r in rows]}
                if want in ('best', 'all'):
                    pick = rows if want == 'all' else [rows[best]]
                    entry['audio'] = [base64.b64encode(open(_encode(r['_path'], 'opus', td, '%s_%d' % (name, r['seed'])), 'rb').read()).decode()
                                      for r in pick]
                if name in results and improved:
                    f = _encode(rows[best]['_path'], 'flac', td, name)
                    layers_io.upload(results[name], f, content_type='audio/flac', deadline=deadline)
                    entry['bytes'] = os.path.getsize(f)
                out[name] = entry
            return {'ok': True, 'audio_s': round(seconds, 2), 'gpu_s': round(gpu_s, 2), 'wall_s': round(time.time() - t0, 2),
                    'gpu': GPU, 'model': model, 'takes': takes, 'tracks': out, 'aes_backing': aes0,
                    'improved': any(v.get('improved') for v in out.values())}
    except layers_io.TransferError as e:
        return {'error': e.code, 'detail': str(e)[:160]}
    except torch.cuda.OutOfMemoryError:
        return {'error': 'gpu_oom'}
    except Exception as e:  # noqa: BLE001 — a crash must reach the app as a code, not a dead job
        return {'error': 'worker_exception', 'detail': (type(e).__name__ + ': ' + str(e))[:240]}


if __name__ == '__main__':
    import runpod
    runpod.serverless.start({'handler': handler})
