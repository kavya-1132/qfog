"""Explainable-AI reasoning engine — same logic as Cell 29 of the notebook."""
import numpy as np
import torch

H = W = 224
_yy, _xx = np.mgrid[0:H, 0:W] / H
BORDER = (_xx < 0.08) | (_xx > 0.92) | (_yy < 0.08) | (_yy > 0.92)
ZONES = {}
for _side, _cond in (('right', _xx < 0.42), ('left', _xx > 0.58)):      # image-left = patient's RIGHT
    for _lvl, (_a, _b) in (('upper', (0.08, 0.37)), ('middle', (0.37, 0.63)), ('lower', (0.63, 0.92))):
        ZONES[f'{_side} {_lvl} zone'] = _cond & (_yy >= _a) & (_yy < _b) & ~BORDER
ZONES['central (heart / mediastinum)'] = (_xx >= 0.42) & (_xx <= 0.58) & ~BORDER
LUNG = [z for z in ZONES if 'zone' in z]
CENTRAL = 'central (heart / mediastinum)'
EXPECTED = {
    'Atelectasis': [z for z in LUNG if 'lower' in z or 'middle' in z], 'Cardiomegaly': [CENTRAL] + [z for z in LUNG if 'lower' in z],
    'Effusion': [z for z in LUNG if 'lower' in z], 'Infiltration': LUNG, 'Mass': LUNG, 'Nodule': LUNG, 'Pneumonia': LUNG,
    'Pneumothorax': [z for z in LUNG if 'upper' in z or 'middle' in z], 'Consolidation': LUNG,
    'Edema': [z for z in LUNG if 'middle' in z] + [CENTRAL], 'Emphysema': [z for z in LUNG if 'upper' in z or 'middle' in z],
    'Fibrosis': [z for z in LUNG if 'lower' in z or 'middle' in z], 'Pleural_Thickening': LUNG,
    'Hernia': [CENTRAL] + [z for z in LUNG if 'lower' in z]}
TYPICAL = {'Atelectasis': 'collapsed lung tissue, often in the lower/middle zones', 'Cardiomegaly': 'an enlarged heart shadow in the central lower chest',
           'Effusion': 'fluid collecting at the lung bases (costophrenic angles)', 'Infiltration': 'patchy opacities anywhere in the lung fields',
           'Mass': 'a focal opacity > 3 cm within the lungs', 'Nodule': 'a small focal opacity within the lungs', 'Pneumonia': 'airspace opacity, often lower zones',
           'Pneumothorax': 'air at the lung apex / periphery', 'Consolidation': 'dense airspace filling of a lung region',
           'Edema': 'bilateral perihilar haziness around the heart', 'Emphysema': 'hyperinflated, darker upper lung zones',
           'Fibrosis': 'reticular scarring, often peripheral and lower', 'Pleural_Thickening': 'thickened pleura along the lung margins',
           'Hernia': 'abdominal content above the diaphragm, low central chest'}


def zone_shares(cam):
    tot = cam.sum() + 1e-8
    out = {z: float((cam * m).sum() / tot) for z, m in ZONES.items()}
    out['image border'] = float((cam * BORDER).sum() / tot)
    return out


def occlusion_test(model, x, cls, cam, frac=0.15):
    """Hide the most-highlighted 15% of pixels; return the new probability (faithfulness check)."""
    m = torch.tensor(cam >= np.quantile(cam, 1 - frac))
    xo = x.clone()
    xo[:, :, m] = 0.0
    with torch.no_grad():
        return float(torch.sigmoid(model(xo))[0, cls])


def reasoning(cls_name, prob, cam, prob_after, att_map=None):
    """Plain-language explanation + structured verdict."""
    sh = zone_shares(cam)
    top_zone = max(ZONES, key=lambda z: sh[z])
    exp = EXPECTED[cls_name]
    exp_share = sum(sh[z] for z in exp)
    exp_area = sum(ZONES[z].mean() for z in exp) / (1 - BORDER.mean())
    lift = exp_share / max(exp_area, 1e-6)
    conf = 'high' if prob >= 0.7 else 'moderate' if prob >= 0.5 else 'low'
    lines = [f'**Finding: {cls_name}** — probability {prob:.2f} ({conf} confidence).',
             f'**Where the model looked:** strongest attention in the **{top_zone}** ({100 * sh[top_zone]:.0f}% of heat). '
             f'{cls_name} typically shows {TYPICAL[cls_name]}.']
    if sh['image border'] >= 0.30:
        verdict = 'possible shortcut'
        lines.append(f'⚠ **Shortcut warning:** {100 * sh["image border"]:.0f}% of the heat lies on the image border (labels, markers, edges), '
                     'not on anatomy. Treat this prediction with caution.')
    elif lift >= 1.15:
        verdict = 'anatomically plausible'
        lines.append(f'✔ **Plausibility:** {100 * exp_share:.0f}% of the heat falls in the zones where {cls_name} is expected '
                     f'({lift:.1f}x more than chance) — the attention is anatomically plausible.')
    else:
        verdict = 'diffuse / uncertain'
        lines.append(f'• **Plausibility:** attention is diffuse ({100 * exp_share:.0f}% in expected zones, {lift:.1f}x chance) — '
                     'the model has not localised a clear region.')
    lines.append(f'**Faithfulness (occlusion test):** hiding the highlighted region changes the probability from {prob:.2f} to {prob_after:.2f} '
                 + ('— the highlighted region genuinely drives the prediction.' if prob - prob_after > 0.05
                    else '— small change; the prediction relies on wider context.'))
    if att_map is not None:
        g = np.asarray(att_map).reshape(7, 7)
        r, c = np.unravel_index(int(np.argmax(g)), g.shape)
        side = "patient's right" if c < 3 else "patient's left" if c > 3 else 'centre'
        lvl = 'upper' if r < 2 else 'lower' if r > 4 else 'middle'
        lines.append(f'**Quantum attention:** the 8-qubit query attended most to the {lvl} {side} region of the image.')
    lines.append('Recommendation: radiologist review advised.' if conf != 'high' or verdict != 'anatomically plausible'
                 else 'Recommendation: finding is confident and well-localised; confirm in routine reporting.')
    return '\n\n'.join(lines), {'verdict': verdict, 'top_zone': top_zone, 'expected_zone_share': exp_share,
                                'border_share': sh['image border'], 'prob_after_occlusion': prob_after, 'zone_shares': sh}
