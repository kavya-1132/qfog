"""Run from the QFog_App folder:  pytest -q   (needs the deploy/ bundle)"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DEPLOY_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(not (DEPLOY_DIR / 'deploy_config.json').exists(), reason='deploy bundle not installed')


@pytest.fixture(scope='module')
def system():
    from src.models import load_bundle
    from src.pipeline import CloudService, FogNode
    cfg, cloud, triage = load_bundle(DEPLOY_DIR)
    return cfg, FogNode(triage, cfg['threshold'], CloudService(cloud, cfg['classes']))


def xray():
    ex = sorted((DEPLOY_DIR / 'examples').glob('*'))
    return Image.open(ex[0]) if ex else Image.fromarray(np.random.randint(0, 255, (300, 300), dtype=np.uint8))


def test_bundle_loads(system):
    cfg, _ = system
    assert len(cfg['classes']) == 14 and cfg['n_qubits'] == 8


def test_cloud_only_returns_diagnosis_and_explanation(system):
    _, fog = system
    r = fog.process(xray(), mode='cloud-only', latency_ms=1, mbps=100)
    assert r['sent_to_cloud'] and 'cloud' in r
    c = r['cloud']
    assert len(c['findings']) == 3 and all(0 <= p <= 1 for _, p in c['findings'])
    assert 'reasoning' in c and c['verdict'] in ('anatomically plausible', 'possible shortcut', 'diffuse / uncertain')


def test_fog_decision_respects_threshold(system):
    cfg, fog = system
    r = fog.process(xray(), mode='fog+cloud', latency_ms=1, mbps=100)
    assert r['sent_to_cloud'] == (r['p_abnormal'] > cfg['threshold'])


def test_cloud_outage_falls_back(system):
    _, fog = system
    r = fog.process(xray(), mode='cloud-only', cloud_down=True)
    assert 'FALLBACK' in r['decision'] and 'cloud' not in r
