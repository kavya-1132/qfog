"""Fog node + cloud service. Same preprocessing, threshold and logic as the notebook.

Transport:
  'in-process' : the cloud service runs inside the app (one command to start; network is emulated)
  'http'       : the cloud service runs as a separate process (python -m uvicorn src.cloud_server:app --port 8001)
"""
import base64
import io
import threading
import time

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .xai import occlusion_test, reasoning

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_gray(src):
    """Path / bytes / PIL image -> 256x256 grayscale uint8."""
    if isinstance(src, (bytes, bytearray)):
        src = Image.open(io.BytesIO(src))
    img = (src if isinstance(src, Image.Image) else Image.open(src)).convert('L').resize((256, 256), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def to_input(gray):
    t = torch.from_numpy(np.array(gray)).float().div(255)[None, None]
    t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)[0]
    return (t.expand(3, -1, -1) - MEAN) / STD


def denormalize(t):
    return (t.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def to_png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def from_png_b64(s):
    return np.array(Image.open(io.BytesIO(base64.b64decode(s))))


class CloudService:
    """Cloud tier: quantum-classical diagnosis + Grad-CAM++ + reasoning."""

    def __init__(self, cloud_model, classes):
        self.model, self.classes = cloud_model, classes
        self.cam = GradCAMPlusPlus(model=cloud_model, target_layers=[cloud_model.backbone.cam_layer])
        self.lock = threading.Lock()

    def diagnose(self, gray, explain=True):
        with self.lock:
            t = {}
            x = to_input(gray)[None]
            t0 = time.perf_counter()
            with torch.no_grad():
                probs = torch.sigmoid(self.model(x))[0].numpy()
                att = self.model.attention_map()[0].numpy()
            t['cloud diagnosis (backbone + 8-qubit VQC + attention)'] = (time.perf_counter() - t0) * 1000
            top = np.argsort(-probs)[:3]
            out = {'findings': [(self.classes[i], float(probs[i])) for i in top],
                   'probs': {c: float(v) for c, v in zip(self.classes, probs)}}
            if explain:
                t0 = time.perf_counter()
                c = int(top[0])
                cam = self.cam(input_tensor=x, targets=[ClassifierOutputTarget(c)])[0]
                after = occlusion_test(self.model, x, c, cam)
                text, info = reasoning(self.classes[c], float(probs[c]), cam, after, att)
                img = denormalize(x[0])
                a = F.interpolate(torch.tensor(att).reshape(1, 1, 7, 7), size=(224, 224), mode='bilinear')[0, 0].numpy()
                a = (a - a.min()) / (a.max() - a.min() + 1e-8)
                out.update({'heatmap_png': to_png_b64(show_cam_on_image(img, cam, use_rgb=True)),
                            'quantum_attention_png': to_png_b64(show_cam_on_image(img, a, use_rgb=True)),
                            'reasoning': text, 'verdict': info['verdict'], 'zone_shares': info['zone_shares'],
                            'prob_after_occlusion': after})
                t['explanation (Grad-CAM++ + reasoning)'] = (time.perf_counter() - t0) * 1000
            out['timings_ms'] = t
            return out


def emulate_link(nbytes, latency_ms, mbps):
    d = latency_ms / 1000 + nbytes * 8 / (mbps * 1e6)
    time.sleep(d)
    return d * 1000


class FogNode:
    """Fog tier: preprocess -> MobileNetV3 triage on CPU -> keep locally or send to the cloud (with fallback)."""

    def __init__(self, triage_model, threshold, cloud_service=None, cloud_url='http://127.0.0.1:8001'):
        self.triage, self.threshold, self.cloud, self.url = triage_model, threshold, cloud_service, cloud_url
        self.adaptive = None   # optional src.adaptive.AdaptiveTriage (audit-driven adaptive threshold); None = fixed threshold

    def process(self, image, mode='fog+cloud', latency_ms=40, mbps=10, cloud_down=False, transport='in-process', reference_abnormal=None):
        rec, t = {'mode': mode, 'transport': transport}, {}
        raw = image if isinstance(image, (bytes, bytearray)) else None
        if raw is None:
            buf = io.BytesIO()
            image.convert('L').save(buf, format='PNG')
            raw = buf.getvalue()
        rec['device_to_fog_bytes'] = len(raw)
        t_all = t0 = time.perf_counter()
        gray = load_gray(raw)
        t['fog preprocess'] = (time.perf_counter() - t0) * 1000
        if mode == 'fog+cloud':
            t0 = time.perf_counter()
            with torch.no_grad():
                p = torch.sigmoid(self.triage(to_input(gray)[None])).item()
            t['fog triage (MobileNetV3, CPU)'] = (time.perf_counter() - t0) * 1000
            rec['p_abnormal'] = p
            if self.adaptive is not None:
                send, audited = self.adaptive.decide(p)
                rule_sent = send and not audited
                rec['threshold_used'] = self.adaptive.threshold
            else:
                send, audited = p > self.threshold, False
                rule_sent = send
                rec['threshold_used'] = self.threshold
            rec['audited'] = audited
        else:
            rec['p_abnormal'], send = None, True
            audited, rule_sent = False, True
            rec['audited'] = False
        rec['sent_to_cloud'] = send
        rec['image'] = denormalize(to_input(gray))
        if not send:
            rec.update(decision='LOW-RISK on the fog screen — not reviewed by the cloud model. Screening result only: '
                                'this is not a diagnosis and not a clearance.', uplink_bytes=0)
        else:
            buf = io.BytesIO()
            Image.fromarray(gray).save(buf, format='JPEG', quality=90)
            payload = buf.getvalue()
            rec['uplink_bytes'] = len(payload)
            try:
                if cloud_down:
                    raise ConnectionError('cloud unreachable (simulated outage)')
                up = emulate_link(len(payload), latency_ms, mbps)
                t0 = time.perf_counter()
                if transport == 'http':
                    r = requests.post(f'{self.url}/diagnose', data=payload, timeout=120)
                    r.raise_for_status()
                    cloud, nbytes = r.json(), len(r.content)
                else:
                    cloud = self.cloud.diagnose(load_gray(payload))
                    nbytes = len(cloud.get('heatmap_png', '')) + len(cloud.get('quantum_attention_png', '')) + 2000
                t['cloud round trip (compute)'] = (time.perf_counter() - t0) * 1000
                t['network (emulated)'] = up + emulate_link(nbytes, latency_ms, mbps)
                rec.update(cloud=cloud, downlink_bytes=nbytes,
                           decision='ABNORMAL — flagged by fog, diagnosed in the cloud' if mode == 'fog+cloud' else 'Diagnosed in the cloud')
            except Exception as e:
                pa = rec['p_abnormal']
                rec.update(decision=f'FALLBACK — {e}. Fog triage p(abnormal) = {pa if pa is not None else float("nan"):.2f}; '
                                    'case queued and flagged for radiologist review.', error=str(e))
        if audited and 'cloud' in rec:
            rec['decision'] = ('AUDIT — the fog rule would have kept this image (low risk); it was sent to the cloud as a random '
                               'quality audit. Reference: ' + rec['decision'])
        if self.adaptive is not None and mode == 'fog+cloud':
            reached = 'cloud' in rec
            self.adaptive.update(rule_sent, audited, reference_abnormal if (reached and reference_abnormal is not None) else None)
            rec['adaptive_state'] = self.adaptive.state()
        t['total'] = (time.perf_counter() - t_all) * 1000
        rec['timings_ms'] = t
        return rec
