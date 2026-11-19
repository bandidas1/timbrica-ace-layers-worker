# Bake the weights into the image, pinned to one Hugging Face revision each (MIT, ACE-Step).
#   ACE-Step/Ace-Step1.5      -> vae/ + Qwen3-Embedding-0.6B/  (the shared VAE and text encoder; the LM and the
#                                turbo DiT in that repo are not needed for lego and are skipped)
#   ACE-Step/acestep-v15-base -> checkpoints/acestep-v15-base/ (the DiT that can do lego)
import os

from huggingface_hub import snapshot_download

ROOT = os.path.join(os.environ.get('ACE_ROOT', '/app/ace'), 'checkpoints')
os.makedirs(ROOT, exist_ok=True)
MAIN_REV = '19671f406d603126926c1b7e2adc169acbcade22'      # ACE-Step/Ace-Step1.5, 19.11.2026
BASE_REV = 'e432212fec32b8965a14ffa57ae653438d6abd14'      # ACE-Step/acestep-v15-base, 19.11.2026

snapshot_download('ACE-Step/Ace-Step1.5', revision=MAIN_REV, local_dir=ROOT,
                  allow_patterns=['vae/*', 'Qwen3-Embedding-0.6B/*', 'config.json', 'configuration.json', 'README.md'])
snapshot_download('ACE-Step/acestep-v15-base', revision=BASE_REV, local_dir=os.path.join(ROOT, 'acestep-v15-base'))
for d in ('vae', 'Qwen3-Embedding-0.6B', 'acestep-v15-base'):
    p = os.path.join(ROOT, d)
    assert os.path.isdir(p) and os.listdir(p), 'missing ' + p
    print(d, sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs) // (1 << 20), 'MB')
