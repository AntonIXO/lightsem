import unittest
import numpy as np
from lightsem.audio.rhythm_clock import RhythmClock,clock_from_audio
from lightsem.experiments.motion_study.companion import (companion_targets,CompanionConfig,compose_companion_world,default_lighting_config)
from lightsem.experiments.motion_study.articulated_stage import NEUTRAL
from lightsem.experiments.motion_study.lighting_map import compose_world_joints
from lightsem.conductor.rig import canonical_rig


def clock(bpm=120,duration=16,confidence=1):
    t=np.arange(round(duration*40)+1)/40
    return RhythmClock(t,t*bpm/60,np.full(len(t),confidence),np.zeros_like(t),np.arange(0,duration,60/bpm))

def body(fps=40,duration=16):
    b=np.broadcast_to(NEUTRAL,(round(fps*duration),22,3)).copy();b[:,:,1]+=.95
    return b

class CompanionTests(unittest.TestCase):
    def test_static_body_can_have_rhythmic_light(self):
        b=body();original=b.copy();target,tr=companion_targets(b,np.full(len(b),.9),clock())
        self.assertGreater(np.std(target[-400:,:,0]),1.)
        self.assertGreater(np.max(np.linalg.norm(np.diff(target[-400:],axis=0),axis=-1)),.05)
        np.testing.assert_array_equal(b,original)

    def test_no_confidence_no_autonomous_motion(self):
        b=body();t,_=companion_targets(b,np.full(len(b),.9),clock(confidence=0))
        np.testing.assert_allclose(t,np.broadcast_to(t[:1],t.shape),atol=1e-12)

    def test_quiet_suppresses_rhythm(self):
        b=body();t,_=companion_targets(b,np.full(len(b),.05),clock())
        np.testing.assert_allclose(t,np.broadcast_to(t[:1],t.shape),atol=1e-12)

    def test_tempo_controls_period_not_body(self):
        b=body(duration=24)
        for bpm in (90,150):
            target,_=companion_targets(b,np.full(len(b),.9),clock(bpm,24))
            # Outer head repeats each four musical beats; use exact-grid120?
            signal=target[:,0,0]-target[:,0,0].mean()
            f=np.fft.rfftfreq(len(signal),1/40)[np.argmax(np.abs(np.fft.rfft(signal))[1:])+1]
            self.assertLess(abs(f-bpm/60/4),1/24+.001)

    def test_phase_shift_changes_targets_without_body_edit(self):
        b=body();a,_=companion_targets(b,np.full(len(b),.9),clock())
        c,_=companion_targets(b,np.full(len(b),.9),clock(),config=CompanionConfig(phase_offset_beats=.5))
        self.assertGreater(np.sqrt(np.mean((a-c)**2)),.2)

    def test_style_changes_shape(self):
        b=body();a,_=companion_targets(b,np.full(len(b),.9),clock())
        b[:,20,1]+=.5
        c,_=companion_targets(b,np.full(len(b),.9),clock())
        self.assertGreater(np.sqrt(np.mean((a-c)**2)),.05)

    def test_emitted_tempo_changes_with_clock(self):
        from scripts.motion.run_rhythm_companion import angles
        duration=24;b=body(30,duration);n=len(b)
        curve={'time':np.arange(n)/30,'momentary':np.full(n,-14.),'short_term':np.full(n,-14.)}
        bundle=compose_world_joints(b,curve,duration,config=default_lighting_config())
        rig=canonical_rig()
        for bpm in (90,150):
            emitted,_,_=compose_companion_world(b,curve,duration,clock(bpm,duration),baseline_bundle=bundle)
            signal=angles(emitted,rig)[:,0,0]
            signal=signal-signal.mean()
            f=np.fft.rfftfreq(len(signal),1/40)[np.argmax(np.abs(np.fft.rfft(signal))[1:])+1]
            self.assertLess(abs(f-bpm/60/4),1/24+.001)

    def test_finite_surface(self):
        b=body();t,_=companion_targets(b,np.ones(len(b)),clock())
        self.assertTrue(np.isfinite(t).all());self.assertLessEqual(abs(t[:,:,0]).max(),7.5)
        self.assertTrue(np.all((abs(t[:,:,1])<1e-10)|(abs(t[:,:,2]+6)<1e-10)))

    def test_clock_validation(self):
        with self.assertRaises(ValueError):RhythmClock([0,1],[1,0],[1,1],[0,0],[])
        with self.assertRaises(ValueError):clock().sample([100])
        with self.assertRaises(ValueError):CompanionConfig(gesture_gain=float('nan'))

    def test_compose_only_changes_angles_and_mechanics(self):
        n=180;duration=n/30;b=body(30,duration);original=b.copy()
        curve={'time':np.arange(n)/30,'momentary':np.full(n,-14.),'short_term':np.full(n,-14.)}
        cfg=default_lighting_config();bundle=compose_world_joints(b,curve,duration,config=cfg)
        frames,trace,report=compose_companion_world(b,curve,duration,clock(duration=duration),baseline_bundle=bundle)
        mask=np.ones(512,bool)
        for f in canonical_rig().by_kind('moving_head'):mask[f.address-1:f.address+3]=False
        np.testing.assert_array_equal(frames[:,mask],bundle[0][:,mask]);np.testing.assert_array_equal(b,original)
        self.assertGreater(np.count_nonzero(frames!=bundle[0]),100)
        for h in report['heads']:
            self.assertLessEqual(h['speed_max_deg_s'],90+1e-7);self.assertLessEqual(h['acceleration_max_deg_s2'],360+1e-7)

class ClockTests(unittest.TestCase):
    def test_silence(self):
        c,r=clock_from_audio(np.zeros(22050*8),22050)
        self.assertEqual(c.confidence.max(),0)

    def check_beats(self,beats,duration=24):
        sr=22050;y=np.zeros(sr*duration)
        pulse=np.sin(2*np.pi*70*np.arange(int(.10*sr))/sr)*np.exp(-np.arange(int(.10*sr))/(.018*sr))
        for beat in beats:
            i=round(beat*sr)
            if i+len(pulse)<=len(y):y[i:i+len(pulse)]+=pulse
        c,report=clock_from_audio(y,sr)
        interior=c.beat_times[(c.beat_times>2)&(c.beat_times<duration-2)]
        error=np.min(abs(interior[:,None]-beats[None]),axis=1)
        print('SYNTHETIC CLOCK',report,'median_error',np.median(error),flush=True)
        self.assertGreater(len(interior),15)
        self.assertLessEqual(np.median(error),.06)
        # Coverage catches half/double-time and accidental single good detection.
        target=beats[(beats>2)&(beats<duration-2)]
        self.assertLessEqual(np.median(np.min(abs(target[:,None]-interior[None]),axis=1)),.06)
        return c

    def test_fixed_tempi(self):
        for bpm in (90,120,150):
            with self.subTest(bpm=bpm):self.check_beats(np.arange(.4,24,60/bpm))

    def test_tempo_ramp(self):
        beats=[];t=.4
        while t<24:beats.append(t);t+=60/(100+40*t/24)
        c=self.check_beats(np.asarray(beats))
        self.assertGreater(np.median(np.diff(c.beat_times[:8])),np.median(np.diff(c.beat_times[-8:])))

if __name__=='__main__':unittest.main()
