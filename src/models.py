"""Model definitions — identical to QFog_Complete.ipynb so the exported weights load exactly."""
import json
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def make_vqc(n, layers):
    """8-qubit data re-uploading circuit with a CNOT ring (PennyLane, simulated)."""
    dev = qml.device('default.qubit', wires=n)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(inputs, weights):
        for l in range(layers):
            qml.AngleEmbedding(inputs, wires=range(n), rotation='Y')
            for i in range(n):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            for i in range(n):
                qml.CNOT(wires=[i, (i + 1) % n])
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return circuit, {'weights': (layers, n, 2)}


class CloudBackbone(nn.Module):
    def __init__(self, arch='densenet121', n_classes=14):
        super().__init__()
        if arch == 'densenet121':
            self.features, self.dim = tvm.densenet121(weights=None).features, 1024
        else:
            r = tvm.resnet18(weights=None)
            self.features = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
            self.dim = 512
        self.fc = nn.Linear(self.dim, n_classes)

    @property
    def cam_layer(self):
        return self.features[-1]

    def tokens(self, x):
        f = F.relu(self.features(x))
        return f.flatten(2).transpose(1, 2), f.mean(dim=(2, 3))

    def forward(self, x):
        return self.fc(self.tokens(x)[1])


class CrossAttentionFusion(nn.Module):
    def __init__(self, nq, dim, d=64):
        super().__init__()
        self.q, self.k, self.v, self.s = nn.Linear(nq, d), nn.Linear(dim, d), nn.Linear(dim, d), d ** -0.5
        self.last_weights = None

    def forward(self, tokens, qf):
        w = torch.softmax(self.q(qf)[:, None] @ self.k(tokens).transpose(1, 2) * self.s, dim=-1)
        self.last_weights = w.detach()
        return (w @ self.v(tokens))[:, 0]


class AttentionHybrid(nn.Module):
    def __init__(self, nq, nl, dim, n_classes):
        super().__init__()
        circ, ws = make_vqc(nq, nl)
        self.branch = qml.qnn.TorchLayer(circ, ws)
        self.fusion = CrossAttentionFusion(nq, dim)
        self.classifier = nn.Sequential(nn.Linear(dim + 64, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, tok, pooled, q):
        return self.classifier(torch.cat([pooled, self.fusion(tok.float(), self.branch(q).float())], 1))


class FrozenPCAScale(nn.Module):
    def __init__(self, nq, dim):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('comp', torch.zeros(nq, dim))
        self.register_buffer('dmin', torch.zeros(nq))
        self.register_buffer('drng', torch.ones(nq))

    def forward(self, x):
        return torch.clamp(((x - self.mean) @ self.comp.T - self.dmin) / self.drng, 0, 1) * np.pi


class CloudModel(nn.Module):
    """Image -> backbone -> frozen PCA -> 8-qubit VQC -> quantum-attention fusion -> 14 logits (seed ensemble)."""

    def __init__(self, bb, pca, heads):
        super().__init__()
        self.backbone, self.pca, self.heads = bb, pca, nn.ModuleList(heads)

    def forward(self, x):
        tok, pooled = self.backbone.tokens(x)
        q = self.pca(pooled)
        return torch.stack([h(tok, pooled, q) for h in self.heads]).mean(0)

    def attention_map(self):
        return torch.stack([h.fusion.last_weights for h in self.heads]).mean(0)[:, 0]


class TriageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = tvm.mobilenet_v3_small(weights=None).features
        self.drop, self.fc = nn.Dropout(0.2), nn.Linear(576, 1)

    def forward(self, x):
        return self.fc(self.drop(self.features(x).mean(dim=(2, 3)))).squeeze(1)


def load_bundle(deploy_dir):
    """Loads deploy_config.json + the exported weights. Returns (cfg, cloud_model, triage_model)."""
    d = Path(deploy_dir)
    cfg = json.loads((d / 'deploy_config.json').read_text())
    n_cls, nq, nl, dim = len(cfg['classes']), cfg['n_qubits'], cfg['n_layers'], cfg['feature_dim']
    bb = CloudBackbone(cfg['backbone'], n_cls)
    heads = [AttentionHybrid(nq, nl, dim, n_cls) for _ in range(cfg['n_heads'])]
    cloud = CloudModel(bb, FrozenPCAScale(nq, dim), heads)
    cloud.load_state_dict(torch.load(d / 'models' / 'cloud_model.pt', map_location='cpu'))
    triage = TriageNet()
    triage.load_state_dict(torch.load(d / 'models' / 'triage.pt', map_location='cpu'))
    return cfg, cloud.eval(), triage.eval()
