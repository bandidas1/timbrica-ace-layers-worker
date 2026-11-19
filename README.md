# timbrica-ace-layers-worker

RunPod serverless worker for «Улучшить минус» (/song-generator): new instrument layers that play along with a
person's backing track (минусовка) — strings, keys, drums, bass… The worker returns the LAYERS only; the page mixes
them over the untouched original.

Engine: ACE-Step 1.5 (github.com/ace-step/ACE-Step-1.5, MIT — code and weights), task `lego`, checkpoint
`acestep-v15-base` (HF rev e432212f…), VAE + Qwen3-Embedding-0.6B from `ACE-Step/Ace-Step1.5` (rev 19671f40…).
ACE code pinned to ca1e85f (29.08.2026, after upstream PR #1287). The 5 Hz LM is not used (thinking off).

Lives under `services/ace-layers-worker/` in the app repo, mirrored to `github.com/bandidas1/timbrica-ace-layers-worker`,
whose CI runs the self-tests and pushes `ghcr.io/bandidas1/timbrica-ace-layers-worker:<commit>`. Pin the endpoint
to the COMMIT TAG.

## Why takes are scored

lego decides by seed what it plays (upstream issue #1375: on a dense mix ~1 usable take in 10). Every track is made
as N takes; `score.py` measures each against the backing — beat (onset cross-correlation), harmony (beat-synchronous
chroma vs the backing, minus the same against the backing rolled ±1..6 semitones), presence, lag — and the best one
is delivered. `score_selftest.py` proves the ranking on synthetic audio in CI.

## Contract

- `{"probe": true}` → `{"ok": true, "probe": true, "gpu": …}`.
- job: `source_url` (signed GET) or `source_b64`; `tracks: [{name, caption}]` (strings, keyboard, drums, bass,
  guitar, synth, brass, woodwinds, percussion); `takes` (≤ 6), `steps`, `guidance`, `shift`, `use_adg`, `bpm`,
  `keyscale`, `model` (`base`; `xl` = bench only, fetched at runtime); `result_urls: {track: signed PUT}` — the best
  take per track, AAC 256k in .m4a; `return_audio: none|best|all` — bench: takes inline as base64 Opus;
  `deadline_at`.
- output: `{"ok", "audio_s", "gpu_s", "wall_s", "gpu", "model", "takes", "tracks": {name: {"best", "takes": [scores]}}}`
  or `{"error": code}` — `no_tracks`, `missing_source`, `source_unreadable`, `too_long`, `deadline_passed`,
  `gpu_oom`, `worker_exception`, `source_*` / `result_*` (transfer).
