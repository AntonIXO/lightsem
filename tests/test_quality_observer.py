import unittest
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from lightsem.experiments.quality_observer import image_pair_motion, phase_explanation, failure_flags, video_observation, beam_observation


class ObserverTests(unittest.TestCase):
    def test_encoded_black_is_not_lit_background(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'black.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=160x90:r=10:d=1',
                            '-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(p)],check=True)
            s=video_observation(p,width=160,height=90)['summary']
            self.assertEqual(s['mean_brightness'],0.)
            self.assertEqual(s['support_fraction'],0.)
            self.assertEqual(s['visible_motion_frame_fraction'],0.)

    def test_small_fast_jitter_does_not_pass_spatial_span(self):
        from lightsem.conductor.rig import canonical_rig
        from lightsem.experiments.beat_conditioned.beat_conditioned_render import controls_to_dmx
        from lightsem.experiments.beat_conditioned.beat_conditioned_data import NAMES
        rig=canonical_rig();frames=controls_to_dmx(np.zeros((400,21)),NAMES,10.)[0]
        for f in rig.fixtures:
            if f.kind!='moving_head':continue
            a=f.address-1;p=frames[:,a].astype(int)*256+frames[:,a+1].astype(int)
            p+=np.rint(20*np.sin(2*np.pi*10*np.arange(400)/40)).astype(int)
            frames[:,a]=(p>>8).astype('uint8');frames[:,a+1]=(p&255).astype('uint8')
        r=beam_observation(frames,rig)
        self.assertGreater(r['lit_motion_fraction'],.9)
        self.assertLess(r['span_2s_median_deg'],1.)

    def image(self):
        y,x=np.mgrid[:90,:160]
        return (200*np.exp(-((x-65)**2/120+(y-45)**2/400))).astype('uint8')

    def test_static_and_global_pulse_are_not_spatial_motion(self):
        a=self.image()
        for gain in [.4,.7,1.,1.2]:
            b=np.clip(a.astype(float)*gain,0,255).astype('uint8')
            r=image_pair_motion(a,b)
            self.assertLess(r['flow_p75_px'],.3)
            self.assertLess(r['shape_change'],.025)

    def test_translating_light_is_detected(self):
        a=self.image();b=np.roll(a,5,axis=1);r=image_pair_motion(a,b)
        self.assertGreater(r['flow_p75_px'],1.)
        self.assertGreater(r['shape_change'],.025)

    def test_phase_template_detects_metronome_not_changing_gestures(self):
        t=np.arange(1600)/20;p=2*np.pi*t*1.5;bar=(t*1.5/4)%1
        r=phase_explanation(np.sin(p),np.sin(p),np.cos(p),bar,fps=20)
        self.assertGreater(r['r2'],.99)
        q=phase_explanation(np.sin(p)*np.where(t<40,1.,.1),np.sin(p),np.cos(p),bar,fps=20)
        self.assertLess(q['r2'],.8)

    def test_rhythmic_but_audio_responsive_is_not_called_beat_collapse(self):
        calibration={'lit_motion_min':.2,'visible_motion_min':.2,'upward_max':.1,
                     'phase_r2_max':.9,'nonbeat_response_min':.1}
        beam={'lit_motion_fraction':.8,'upward_light_fraction':0.}
        screen={'visible_motion_frame_fraction':.7}
        self.assertTrue(failure_flags(beam,screen,.99,.01,calibration=calibration)['beat_shortcut'])
        self.assertFalse(failure_flags(beam,screen,.99,.4,calibration=calibration)['beat_shortcut'])


if __name__=='__main__':unittest.main()
