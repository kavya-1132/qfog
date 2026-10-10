"""QFog — Quantum-Enhanced Fog Learning: clinical decision-support website.
Run from the QFog_App folder:   streamlit run app/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DEPLOY_DIR  # noqa: E402

st.set_page_config(page_title='QFog — Quantum-Enhanced Fog Learning', page_icon='🫁', layout='wide')
st.markdown(f"<style>{(ROOT / 'app/assets/style.css').read_text()}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------- loading
if not (DEPLOY_DIR / 'deploy_config.json').exists():
    st.error('Deployment bundle not found.')
    st.markdown(f"""
1. Run **QFog_Complete.ipynb** in Colab (Run all). Cell 41 writes `MyDrive/qfog_project/qfog_deploy.zip`.
2. Download that zip and **unzip it into** `{DEPLOY_DIR}` so that `{DEPLOY_DIR / 'deploy_config.json'}` exists.
3. Restart this app: `streamlit run app/app.py`.""")
    st.stop()


@st.cache_resource(show_spinner='Loading the fog and cloud models...')
def get_system():
    from src.models import load_bundle
    from src.pipeline import CloudService, FogNode
    cfg, cloud_model, triage = load_bundle(DEPLOY_DIR)
    cloud = CloudService(cloud_model, cfg['classes'])
    return cfg, cloud, FogNode(triage, cfg['threshold'], cloud)


@st.cache_data
def table(name, index_col=0):
    p = DEPLOY_DIR / 'results' / name
    return pd.read_csv(p, index_col=index_col) if p.exists() else pd.DataFrame()


def fig(name, caption=None):
    p = DEPLOY_DIR / 'figures' / name
    if p.exists():
        st.image(str(p), caption=caption, width='stretch')


cfg, cloud_service, fog = get_system()
summary = table('summary.csv')
comparison = table('model_comparison.csv')

with st.sidebar:
    st.title('🫁 QFog')
    page = st.radio('Project sections', ['Dashboard', 'Dataset Explorer', 'Exploratory Data Analysis', 'Quantum Design (16 → 8 qubits)',
                                         'Model Performance', 'Fog–Cloud System', 'Explainable AI', 'Live X-ray Diagnosis'])
    st.caption(f"Cloud: {cfg['backbone']} + {cfg['n_qubits']}-qubit VQC · Fog: MobileNetV3-Small (CPU)")
    st.caption('Local research prototype · no external APIs')

st.markdown('<div class="hero"><h1>Quantum-Enhanced Fog Learning for Early Pulmonary Disorder Detection</h1>'
            '<p>A hybrid quantum–classical, fog–cloud clinical decision-support system with explainable AI.</p></div>', unsafe_allow_html=True)


def cards(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(f'<div class="card"><b>{label}</b><h2>{value}</h2></div>', unsafe_allow_html=True)


DISCLAIMER = ('<div class="disclaimer"><b>Research prototype disclaimer:</b> outputs are model estimates from a public benchmark dataset. '
              'They are not a diagnosis, medical advice, or a substitute for a radiologist.</div>')

# ------------------------------------------------------------------------------------------- pages
if page == 'Dashboard':
    st.subheader('Research dashboard')
    ft = cfg['fog_test']
    cards([('X-ray images', f"{cfg['dataset']['images']:,}"), ('Patients', f"{cfg['dataset']['patients']:,}"),
           ('Findings', cfg['dataset']['classes']), ('Qubits', cfg['n_qubits']),
           ('Fog recall', f"{ft['recall (abnormal sent to cloud)']:.0%}"), ('Cloud calls avoided', f"{cfg['savings']['cloud calls avoided %']:.0f}%")])
    st.write('')
    st.markdown('**Workflow:** X-ray → fog preprocessing → fog triage (CPU) → *if abnormal:* network → cloud backbone → PCA (8) → '
                '8-qubit variational quantum circuit → quantum-attention fusion → 14 findings → Grad-CAM++ & reasoning → clinical report')
    a, b = st.columns([3, 2])
    with a:
        st.markdown('**Key results (computed by the notebook)**')
        if not summary.empty:
            st.dataframe(summary, width='stretch')
    with b:
        st.markdown('**System architecture**')
        st.code('''X-ray device
   │  PNG
   ▼
FOG NODE (CPU) ── low-risk ──▶ local screening note (not reviewed by the cloud)
   │ abnormal
   ▼  JPEG over network
CLOUD TIER
   DenseNet121 → PCA(8) → 8-qubit VQC
   → quantum attention → 14 findings
   → Grad-CAM++ + reasoning
   │  JSON report
   ▼
Clinician (fog fallback if cloud is down)''', language=None)
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

elif page == 'Dataset Explorer':
    st.subheader('Dataset Explorer — NIH ChestX-ray14 (sample)')
    st.write('Frontal chest X-rays from the NIH Clinical Center with 14 labels mined from radiology reports (multi-label; "No Finding" = normal).')
    a, b = st.columns(2)
    with a:
        st.markdown('**Dataset audit**'); st.dataframe(table('dataset_audit.csv'), width='stretch')
    with b:
        st.markdown('**Patient-level split** (no patient appears in two sets)'); st.dataframe(table('split_table.csv'), width='stretch')
    fig('eda_examples.png', 'One example X-ray per class')
    with st.expander('Preprocessing'):
        st.write('Images are converted to grayscale, resized to 256×256 (cache) and 224×224 (model input), ImageNet-normalised. '
                 'Training augmentation: small crop, ±7° rotation, brightness/contrast. No horizontal flip (it would move the heart).')
        fig('augmentation.png')

elif page == 'Exploratory Data Analysis':
    st.subheader('Exploratory Data Analysis')
    t1, t2, t3 = st.tabs(['Class distribution', 'Co-occurrence & demographics', 'Per-class test positives'])
    with t1:
        fig('eda_class_distribution.png')
    with t2:
        fig('eda_cooccurrence_demographics.png')
    with t3:
        pc = table('per_class_auroc.csv')
        if 'test positives' in pc:
            st.plotly_chart(px.bar(pc.reset_index(), x=pc.index.name or 'index', y='test positives', color='reliable',
                                   title='Positives per class in the test set (classes with < 15 are statistically unreliable)'),
                            width='stretch')

elif page == 'Quantum Design (16 → 8 qubits)':
    st.subheader('Why 8 qubits and not 16?')
    st.markdown('The cloud feature vector is compressed by **PCA** to *n* values and encoded as rotation angles on *n* qubits '
                '(3 layers of data re-uploading, trainable RY/RZ rotations, CNOT ring). We measured what happens as *n* grows:')
    qs = table('qubit_study.csv')
    if not qs.empty:
        q = qs.reset_index()
        c = st.columns(3)
        c[0].plotly_chart(px.line(q, x='qubits', y='output spread between patients (std)', markers=True, title='Output spread between patients'), width='stretch')
        c[1].plotly_chart(px.line(q, x='qubits', y='gradient variance', markers=True, log_y=True, title='Gradient variance (barren plateau)'), width='stretch')
        c[2].plotly_chart(px.line(q, x='qubits', y='train step time, batch 16 (s)', markers=True, log_y=True, title='Training step time'), width='stretch')
        st.dataframe(qs.round(4), width='stretch')
    h2h = table('qubit_head_to_head.csv')
    if not h2h.empty:
        st.markdown('**Same training budget, 8 vs 16 qubits**'); st.dataframe(h2h, width='stretch')
    st.success('**Decision:** with more qubits the outputs bunch together across patients, gradients vanish (barren plateau) and the simulated '
               'state doubles in size per qubit (256 amplitudes at 8 qubits vs 65,536 at 16). 8 qubits is the largest circuit that stays trainable and affordable.')
    fig('qubit_study.png', 'Full qubit study (from the notebook)')

elif page == 'Model Performance':
    st.subheader('Model performance (patient-level held-out test set)')
    if not comparison.empty:
        st.dataframe(comparison.round(4), width='stretch')
    st.markdown('**Significance tests** (paired bootstrap, 95% CI). The *classical control* is identical to the attention hybrid except that the '
                'quantum circuit is replaced by a classical layer, so it isolates the quantum contribution.')
    st.dataframe(table('significance_tests.csv'), width='stretch')
    pc = table('per_class_auroc.csv')
    if not pc.empty:
        cols = [c for c in pc.columns if c not in ('test positives', 'reliable')]
        long = pc[pc['reliable'] == True][cols].reset_index().melt(id_vars=pc.index.name or 'index', var_name='model', value_name='AUROC')  # noqa: E712
        st.plotly_chart(px.bar(long, x=pc.index.name or 'index', y='AUROC', color='model', barmode='group', range_y=[0.4, 1.0],
                               title='Per-class AUROC (reliable classes)'), width='stretch')
    a, b = st.columns(2)
    with a: fig('baseline_roc.png', 'Cloud baseline ROC curves')
    with b: fig('backbone_loss.png', 'Backbone training curve')
    st.caption('AUROC = probability that a random positive case is scored higher than a random negative case (0.5 = chance).')

elif page == 'Fog–Cloud System':
    st.subheader('Fog–cloud system')
    ft, sv = cfg['fog_test'], cfg['savings']
    cards([('Fog threshold (from validation)', cfg['threshold']), ('Recall: abnormal sent to cloud', f"{ft['recall (abnormal sent to cloud)']:.0%}"),
           ('Normals kept at fog', f"{ft['specificity (normals kept at fog)']:.0%}"), ('Fog CPU speed-up', f"{cfg['speedup']:.1f}x"),
           ('Upload saved', f"{sv['upload saved %']:.0f}%")])
    st.write('')
    fig('fog_triage.png', 'Fog triage: ROC, validation threshold sweep, test confusion matrix')
    sysdf = table('system_experiment.csv')
    if not sysdf.empty:
        st.markdown('**Running system** — fog+cloud vs cloud-only on different networks, and during a cloud outage')
        st.dataframe(sysdf, width='stretch')
        s = sysdf[~sysdf.index.str.contains('outage')].reset_index().rename(columns={sysdf.index.name or 'index': 'run'})
        st.plotly_chart(px.bar(s, x='run', y='mean latency ms', color=s['run'].str.split(' \\|').str[0], title='Mean end-to-end latency'),
                        width='stretch')
    st.markdown('**Projected savings when most scans are normal (screening setting)**')
    st.dataframe(table('prevalence_projection.csv'), width='stretch')

elif page == 'Explainable AI':
    st.subheader('Explainable AI')
    st.markdown('| Question | Method |\n|---|---|\n| Where did the model look? | Grad-CAM++ through the full hybrid (gradients pass through the quantum circuit) |'
                '\n| What did the quantum part focus on? | Quantum-attention map over 49 image regions |'
                '\n| Does it make anatomical sense? | Zone analysis + plausibility check; shortcut warning if heat sits on the image border |'
                '\n| Is the explanation faithful? | Occlusion test: hide the highlighted pixels and measure the probability drop |')
    fig('xai_gradcam_quantum_attention.png', 'Rows: X-ray · Grad-CAM++ baseline · Grad-CAM++ hybrid · quantum attention')
    xs = cfg.get('xai_summary', {})
    if xs:
        cards([('Anatomically plausible', f"{xs.get('anatomically plausible %', 0):.0f}%"), ('Possible shortcut', f"{xs.get('possible shortcut %', 0):.0f}%"),
               ('Heat on border', f"{xs.get('mean heat on border %', 0):.0f}%"),
               ('Prob. drop: highlighted vs random', f"{xs.get('prob. drop, highlighted pixels removed', 0):.2f} vs {xs.get('prob. drop, random pixels removed', 0):.2f}")])
    a, b = st.columns(2)
    with a: fig('xai_global.png', 'Verdicts and faithfulness over test images')
    with b: fig('xai_qubit_attribution.png', 'Which qubit inputs drive each finding')
    with st.expander('Limitations of the explanations'):
        st.write('Heatmaps show which regions influenced the model, not medical causation. The zone-based plausibility check is a simple '
                 'anatomical heuristic, not a radiologist. Labels in this dataset are mined from reports and contain noise.')

elif page == 'Live X-ray Diagnosis':
    st.subheader('Live X-ray diagnosis')
    with st.sidebar:
        st.markdown('---\n**Demo settings**')
        mode = st.radio('System mode', ['fog+cloud', 'cloud-only'])
        transport = st.radio('Cloud connection', ['in-process', 'HTTP (separate cloud process)'],
                             help='HTTP needs: python -m uvicorn src.cloud_server:app --port 8001')
        latency = st.slider('Network latency (ms)', 5, 300, 40, 5)
        bandwidth = st.slider('Bandwidth (Mbps)', 0.5, 100.0, 10.0)
        outage = st.checkbox('Simulate cloud outage')
    transport = 'http' if transport.startswith('HTTP') else 'in-process'
    if transport == 'http':
        try:
            ok = requests.get('http://127.0.0.1:8001/health', timeout=1).ok
        except Exception:
            ok = False
        st.info('Cloud service on port 8001: ' + ('connected ✔' if ok else 'not running — start it with `python -m uvicorn src.cloud_server:app --port 8001`'))

    examples = sorted((DEPLOY_DIR / 'examples').glob('*'))
    labels = {'— upload my own —': None}
    for p in examples:
        kind, lab, _ = (p.name.split('__') + ['', ''])[:3]
        labels[f'{kind.capitalize()} example · {lab.replace("+", ", ")}'] = p
    choice = st.selectbox('Choose an example X-ray from the test set, or upload your own', list(labels))
    up = st.file_uploader('Upload a frontal chest X-ray (PNG / JPG)', type=['png', 'jpg', 'jpeg'])
    img = Image.open(up) if up is not None else (Image.open(labels[choice]) if labels[choice] else None)

    if img is not None and st.button('Run diagnosis', type='primary'):
        with st.spinner('Fog node → network → cloud tier → explanation...'):
            rec = fog.process(img, mode=mode, latency_ms=latency, mbps=bandwidth, cloud_down=outage, transport=transport)
        sent, ok = rec['sent_to_cloud'], 'cloud' in rec
        steps = [('X-ray received', 'done'), ('Fog triage', 'done' if mode == 'fog+cloud' else 'skip'),
                 ('Network', 'done' if ok else ('fail' if sent else 'skip')), ('Quantum-classical diagnosis', 'done' if ok else 'skip'),
                 ('Grad-CAM++ & reasoning', 'done' if ok else 'skip'), ('Report', 'done')]
        st.markdown('<div class="flow">' + '<span class="arrow">→</span>'.join(f'<span class="step {s}">{n}</span>' for n, s in steps) + '</div>',
                    unsafe_allow_html=True)
        css = 'decision-normal' if not sent else ('decision-abnormal' if ok else 'decision-fallback')
        pa = rec['p_abnormal']
        st.markdown(f'<div class="{css}"><b>Fog node:</b> ' + (f'p(abnormal) = {pa:.2f} (threshold {cfg["threshold"]}) · ' if pa is not None else 'cloud-only mode · ')
                    + f'<b>{rec["decision"]}</b></div>', unsafe_allow_html=True)
        st.write('')
        c1, c2, c3 = st.columns(3)
        c1.image(rec['image'], caption='Input X-ray (as the model sees it)', width='stretch')
        if ok:
            from src.pipeline import from_png_b64
            cl = rec['cloud']
            c2.image(from_png_b64(cl['heatmap_png']), caption='Grad-CAM++ through the quantum hybrid', width='stretch')
            c3.image(from_png_b64(cl['quantum_attention_png']), caption='Quantum attention map', width='stretch')
            a, b = st.columns([1, 1])
            with a:
                st.markdown('#### Explainable-AI reasoning')
                st.markdown(cl['reasoning'])
            with b:
                probs = pd.DataFrame(sorted(cl['probs'].items(), key=lambda kv: kv[1]), columns=['finding', 'probability'])
                st.plotly_chart(px.bar(probs, x='probability', y='finding', orientation='h', range_x=[0, 1], title='All 14 findings'),
                                width='stretch')
                zs = pd.DataFrame(sorted(cl['zone_shares'].items(), key=lambda kv: kv[1]), columns=['zone', 'share of heat'])
                st.plotly_chart(px.bar(zs, x='share of heat', y='zone', orientation='h', title='Where the heat falls'), width='stretch')
        else:
            c2.info('No cloud call was needed.' if not sent else 'Cloud unreachable — fog fallback used.')
        st.markdown('#### Timing and data')
        tm = pd.DataFrame(list(rec['timings_ms'].items()), columns=['stage', 'ms']).round(1)
        st.dataframe(tm, width='stretch', hide_index=True)
        st.caption(f"Data sent to cloud: {rec.get('uplink_bytes', 0) / 1000:.1f} KB (original image {rec['device_to_fog_bytes'] / 1000:.0f} KB)")
        st.markdown(DISCLAIMER, unsafe_allow_html=True)
    elif img is not None:
        st.image(img, caption='Selected X-ray', width=320)
