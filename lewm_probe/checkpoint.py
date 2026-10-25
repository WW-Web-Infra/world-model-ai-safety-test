"""Load the LeWM TwoRooms checkpoint from a pinned Hugging Face revision, verified by SHA-256.

The checkpoint's config.json is turned into a model with Hydra's `instantiate`, which imports and
calls whatever each `_target_` names. To keep a changed or tampered upstream file from running
code here, this module:
  - downloads only from a fixed revision of the Hugging Face repo, never from `main`
  - checks the SHA-256 of config.json and weights.pt on every load, not only after download
  - accepts only the `_target_` classes this checkpoint is known to use, and no Hydra keys
    other than `_target_` and `_partial_`
  - loads the weights with `torch.load(weights_only=True)`, which refuses pickled code
"""

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

import torch
from hydra.utils import instantiate

REPO = 'quentinll/lewm-tworooms'
REVISION = '77adaae0bc31deab21c93740d1f8bb947cd0bdec'
SHA256 = {
    'config.json': '2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09',
    'weights.pt': '566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd',
}
ALLOWED_TARGETS = {
    'stable_worldmodel.wm.lewm.LeWM',
    'stable_worldmodel.wm.lewm.module.Predictor',
    'stable_worldmodel.wm.lewm.module.Embedder',
    'stable_worldmodel.wm.lewm.module.MLP',
    'stable_pretraining.backbone.utils.vit_hf',
    'torch.nn.BatchNorm1d',
}
ALLOWED_HYDRA_KEYS = {'_target_', '_partial_'}

# the dataset archive (quentinll/lewm-tworooms dataset repo), for verifying a manual download
DATASET_REVISION = '6903a2de048b13819d812da0b4dd661290bc01e4'
DATASET_SHA256 = {'tworoom.tar.zst': '494b1a02f0765cd9a0d9daf1786c419ced1009977fc45d01e3158932f8d080ca'}


class IntegrityError(RuntimeError):
    pass


def cache_dir():
    root = Path(os.environ.get('STABLEWM_HOME', Path.home() / '.stable-wm'))
    return root / 'checkpoints' / f'models--{REPO.replace("/", "--")}'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(name):
    """Path to a verified checkpoint file, downloading it from the pinned revision if missing."""
    path = cache_dir() / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            with urllib.request.urlopen(url, timeout=60) as r:
                for chunk in iter(lambda: r.read(1 << 20), b''):
                    tmp.write(chunk)
        if sha256(tmp.name) != SHA256[name]:
            os.unlink(tmp.name)
            raise IntegrityError(f'{name} from {url} does not match its pinned SHA-256')
        os.replace(tmp.name, path)
    if sha256(path) != SHA256[name]:
        raise IntegrityError(f'{path} does not match its pinned SHA-256; delete it to re-download')
    return path


def check_config(node, where='config'):
    """Raise unless every Hydra key and `_target_` in the config is on the allowlist."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key.startswith('_') and key.endswith('_'):
                if key not in ALLOWED_HYDRA_KEYS:
                    raise IntegrityError(f'{where}: Hydra key {key!r} is not allowed')
                if key == '_target_' and value not in ALLOWED_TARGETS:
                    raise IntegrityError(f'{where}: _target_ {value!r} is not allowed')
            check_config(value, f'{where}.{key}')
    elif isinstance(node, list):
        for i, value in enumerate(node):
            check_config(value, f'{where}[{i}]')


def config():
    cfg = json.loads(fetch('config.json').read_text())
    check_config(cfg)
    return cfg


def build_untrained(seed=0):
    """The checkpoint's architecture with default initialisation (the random-init control)."""
    torch.manual_seed(seed)
    return instantiate(config()).eval().requires_grad_(False)


def load():
    """The trained checkpoint, in eval mode with gradients off."""
    model = instantiate(config())
    state = torch.load(fetch('weights.pt'), map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)
