"""Launchers execute only a stub uv process, never a robot or policy."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def launch(tmp_path):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in ('deployment_options', 'calibrate_wrist_camera', 'deploy_policy', 'serve_policy'):
        shutil.copyfile(ROOT / 'scripts' / f'{name}.sh', scripts / f'{name}.sh')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    stub = bindir / 'uv'
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    stub.chmod(0o755)
    (tmp_path / '.env').write_text('EMBODIMENT=sharpa\nWRIST_CAMERA_ID=123\nEXTERNAL_CAMERA_ID=456\nCLOAK_NUC_IP=fixture\nCHECKPOINT_DIR="/fixture/checkpoint with spaces"\nPOLICY_HOST=fixture\nPOLICY_PORT=8123\nDEBUG_MASKS=true\n')
    def run(name, args=(), settings=None):
        env = {k:v for k,v in os.environ.items() if k not in 'EMBODIMENT WRIST_CAMERA_ID EXTERNAL_CAMERA_ID WRIST_CAM_LENS CLOAK_NUC_IP CHECKPOINT_DIR CONFIG POLICY_HOST POLICY_PORT DEBUG_MASKS DEBUG TIMESTEPS'.split()}
        env.update(settings or {})
        env['PATH'] = str(bindir) + os.pathsep + env['PATH']
        return subprocess.run(['bash', str(scripts / f'{name}.sh'), *args], env=env, text=True, capture_output=True)
    return run

@pytest.mark.parametrize('name', ['calibrate_wrist_camera','serve_policy','deploy_policy'])
@pytest.mark.parametrize('embodiment', ['robotiq','umi','sharpa','yam'])
def test_cli_overrides_env_and_preserves_arguments(launch, name, embodiment):
    result=launch(name,['--embodiment',embodiment],{'EMBODIMENT':'umi'})
    assert result.returncode == 0, result.stderr
    args=result.stdout.splitlines()
    assert args[args.index('--embodiment')+1] == embodiment
    if name=='serve_policy':
        assert ('--fixed_hand_sharpa_ik' in args) == (embodiment=='sharpa')
        assert '--debug-masks' in args
        assert '--policy.dir=/fixture/checkpoint with spaces' in args

@pytest.mark.parametrize('name', ['calibrate_wrist_camera','serve_policy','deploy_policy'])
def test_environment_overrides_dotenv(launch,name):
    r=launch(name,settings={'EMBODIMENT':'robotiq'})
    assert r.returncode==0
    args=r.stdout.splitlines();assert args[args.index('--embodiment')+1]=='robotiq'

@pytest.mark.parametrize('args',[['--embodiment'],['--embodiment','invalid']])
def test_invalid_embodiment_rejected(launch,args):
    assert launch('deploy_policy',args).returncode != 0

def test_client_endpoint_and_lens(launch):
    r=launch('deploy_policy',settings={'POLICY_HOST':'override','POLICY_PORT':'9001','WRIST_CAM_LENS':'right','TIMESTEPS':'17'})
    assert r.returncode==0
    a=r.stdout.splitlines()
    for key,value in [('--remote-host','override'),('--remote-port','9001'),('--wrist-cam-lens','right'),('--max-timesteps','17')]:
        assert a[a.index(key)+1]==value

def test_offline_capture_requires_no_camera(launch):
    r=launch('calibrate_wrist_camera',['--captures','/fixture/captures with spaces'],{'WRIST_CAMERA_ID':''})
    assert r.returncode==0
    assert r.stdout.splitlines()[-1]=='/fixture/captures with spaces'

@pytest.mark.parametrize('setting',[{'DEBUG':'invalid'},{'TIMESTEPS':'-1'},{'WRIST_CAMERA_ID':''},{'CLOAK_NUC_IP':''}])
def test_live_client_settings_rejected(launch,setting):
    assert launch('deploy_policy',settings=setting).returncode != 0

def test_mask_flag_can_be_disabled(launch):
    r=launch('serve_policy',settings={'DEBUG_MASKS':'false'})
    assert r.returncode==0
    assert '--debug-masks' not in r.stdout.splitlines()
