import torch

from lightsem.experiments.audio_light_foundation import AudioToLightFoundation, AudioToLightConfig
from lightsem.experiments.audio_light_foundation.data import make_fixture_metadata

def _inputs(frames=64, rate=30):
    g=torch.Generator().manual_seed(7)
    model=AudioToLightFoundation(AudioToLightConfig(local_rate_hz=rate)).eval()
    audio=torch.randn(1,frames,1116,generator=g)
    fixtures=torch.from_numpy(make_fixture_metadata())[None]
    timestamps=torch.arange(frames).float()[None]/rate
    return model,audio,fixtures,timestamps

def test_audio_only_forward_has_no_body_input_and_is_finite():
    model,audio,fixtures,timestamps=_inputs(16)
    with torch.no_grad(): out=model(audio,fixtures,timestamps=timestamps,duration_seconds=16/30)
    assert out['aim'].shape==(1,16,8,2)
    assert out['rgb'].shape==(1,16,8,3)
    assert all(torch.isfinite(v).all() for v in out.values() if torch.is_tensor(v))

def test_declared_32_frame_streaming_contract_matches_full():
    model,audio,fixtures,timestamps=_inputs(64)
    with torch.no_grad():
        full=model(audio,fixtures,timestamps=timestamps,duration_seconds=64/30)
        state=None; pieces=[]
        for start in range(0,64,32):
            part=model(audio[:,start:start+32],fixtures,timestamps=timestamps[:,start:start+32],state=state,duration_seconds=64/30)
            state=part['state']; pieces.append(part)
        for key in ('aim','rgb','brightness','event'):
            assert torch.allclose(full[key],torch.cat([x[key] for x in pieces],1),atol=1e-5,rtol=1e-5), key

def test_streaming_contract_rejects_unaligned_claim_by_measurement():
    model,audio,fixtures,timestamps=_inputs(64)
    with torch.no_grad():
        full=model(audio,fixtures,timestamps=timestamps,duration_seconds=64/30)
        state=None; pieces=[]
        for start in range(0,64,16):
            part=model(audio[:,start:start+16],fixtures,timestamps=timestamps[:,start:start+16],state=state,duration_seconds=64/30)
            state=part['state']; pieces.append(part)
        err=(full['rgb']-torch.cat([x['rgb'] for x in pieces],1)).abs().max()
    assert float(err)>1e-5
