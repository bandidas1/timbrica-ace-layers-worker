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
#    "result_urls": {"strings": signed PUT, ...},         production: the best take per track, AAC in .m4a
#    "return_audio": "best" | "all" | "none",             bench: takes inline as base64 Opus
#    "deadline_at": unix seconds}
# Output {"ok": true, "audio_s", "gpu_s", "gpu", "model", "tracks": {name: {"best": i, "takes": [score...]}}} or
#        {"error": code}
import base64
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
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
    if fmt == 'opus':
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
                    s['seed'] = seed_base + k
                    s['_path'] = path
                    rows.append(s)
                best = int(np.argmax([r['rank'] for r in rows]))
                entry = {'best': best, 'takes': [{k: v for k, v in r.items() if k != '_path'} for r in rows]}
                if want in ('best', 'all'):
                    pick = rows if want == 'all' else [rows[best]]
                    entry['audio'] = [base64.b64encode(open(_encode(r['_path'], 'opus', td, '%s_%d' % (name, r['seed'])), 'rb').read()).decode()
                                      for r in pick]
                if name in results:
                    f = _encode(rows[best]['_path'], 'aac', td, name)
                    layers_io.upload(results[name], f, content_type='audio/mp4', deadline=deadline)
                    entry['bytes'] = os.path.getsize(f)
                out[name] = entry
            return {'ok': True, 'audio_s': round(seconds, 2), 'gpu_s': round(gpu_s, 2), 'wall_s': round(time.time() - t0, 2),
                    'gpu': GPU, 'model': model, 'takes': takes, 'tracks': out}
    except layers_io.TransferError as e:
        return {'error': e.code, 'detail': str(e)[:160]}
    except torch.cuda.OutOfMemoryError:
        return {'error': 'gpu_oom'}
    except Exception as e:  # noqa: BLE001 — a crash must reach the app as a code, not a dead job
        return {'error': 'worker_exception', 'detail': (type(e).__name__ + ': ' + str(e))[:240]}


if __name__ == '__main__':
    import runpod
    runpod.serverless.start({'handler': handler})
