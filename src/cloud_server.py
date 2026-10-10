"""Stand-alone cloud tier as a web service (optional — shows fog and cloud as two separate processes).

Run from the QFog_App folder:   python -m uvicorn src.cloud_server:app --port 8001
Then choose "HTTP (separate cloud process)" in the app's sidebar.
"""
from fastapi import FastAPI, Request

from .config import DEPLOY_DIR
from .models import load_bundle
from .pipeline import CloudService, load_gray

cfg, cloud_model, _ = load_bundle(DEPLOY_DIR)
service = CloudService(cloud_model, cfg['classes'])
app = FastAPI(title='QFog cloud tier')


@app.get('/health')
def health():
    return {'ok': True, 'model': f"{cfg['backbone']} + {cfg['n_qubits']}-qubit VQC attention hybrid"}


@app.post('/diagnose')
async def diagnose(request: Request):
    return service.diagnose(load_gray(await request.body()))
