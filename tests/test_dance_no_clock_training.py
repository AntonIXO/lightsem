"""Focused CPU unit checks; no scientific CPU training or model inference."""
from io import BytesIO
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lightsem.experiments import dance_no_clock_training as training
from lightsem.experiments.stage11_common import rng_state, restore_rng, json_digest


def scales():
    return {'marginal_iqr':[1.]*20,'lag_iqr':{str(lag):[float(lag)]*20 for lag in training.LAGS}}


class DanceNoClockTests(unittest.TestCase):
    def test_twenty_channel_metrics_include_last_channel_and_identity_null(self):
        x=np.random.default_rng(0).normal(size=(200,20))
        same=training.continuous_metrics(x,x,scales())
        self.assertEqual(same['temporal_w1'],0)
        self.assertEqual(same['marginal_w1'],0)
        self.assertEqual(same['variation_ratio'],1)
        self.assertEqual(same['hf_ratio'],1)
        y=x.copy();y[:,19]=0
        different=training.continuous_metrics(x,y,scales())
        self.assertGreater(different['temporal_w1'],0)
        self.assertGreater(different['marginal_w1'],0)

    def test_variation_uses_all_lags_and_training_lag_iqr(self):
        x=np.arange(200,dtype=float)[:,None]*np.ones((1,20))
        result=training.continuous_metrics(x,x*.5,scales())
        self.assertAlmostEqual(result['reference_variation'],1.)
        self.assertAlmostEqual(result['generated_variation'],.5)
        self.assertAlmostEqual(result['variation_ratio'],.5)

    def test_alignment_uses_identical_valid_centered_one_second_windows(self):
        truth=np.zeros((100,20));truth[20:40,0]=1;truth[40:60,1]=2
        wrong=np.roll(truth,20,axis=0)
        result=training.alignment_mse(truth,truth,wrong,[0,1])
        self.assertEqual(result['correct_mse'],0)
        self.assertGreater(result['wrong_mse'],0)
        self.assertEqual(result['improvement_fraction'],1)
        self.assertEqual(result['smoothed_frames'],91)
        self.assertEqual(result['smoothing_frames'],10)
        self.assertEqual(result['smoothing_center_offset_frames'],4.5)

    def test_zero_wrong_audio_error_is_explicitly_ineligible(self):
        x=np.ones((50,20))
        result=training.alignment_mse(x,x,x,[0,1])
        self.assertFalse(result['eligible'])
        self.assertIsNone(result['improvement_fraction'])

    def test_donor_is_next_canonical_song_and_first_recording_coach(self):
        entries=[{'canonical_id':'b','recording_id':'b2','key':'z'},
                 {'canonical_id':'a','recording_id':'a1','key':'b'},
                 {'canonical_id':'b','recording_id':'b1','key':'b'},
                 {'canonical_id':'b','recording_id':'b1','key':'a'},
                 {'canonical_id':'c','recording_id':'c1','key':'c'}]
        mapping=training.donor_map(entries)
        self.assertEqual(mapping['a']['key'],'a')
        self.assertEqual(mapping['b']['canonical_id'],'c')
        self.assertEqual(mapping['c']['canonical_id'],'a')
        self.assertTrue(all(song!=entry['canonical_id'] for song,entry in mapping.items()))

    @staticmethod
    def metric_row(song,recording,ref,generated,correct,wrong):
        return {'canonical_id':song,'recording_id':recording,'metrics':{
            'temporal_w1':.2,'marginal_w1':.3,'reference_variation':ref,'generated_variation':generated,
            'reference_hf_energy':ref,'generated_hf_energy':generated},
            'alignment':{'eligible':True,'correct_mse':correct,'wrong_mse':wrong,
                         'improvement_fraction':(wrong-correct)/wrong}}

    def test_hierarchical_mse_precedes_fraction_and_ratios_are_averaged_per_song(self):
        rows=[self.metric_row('a','a1',1,1,1,2),self.metric_row('a','a1',1,1,9,10),
              self.metric_row('b','b1',10,20,2,4)]
        result=training.summarize(rows)
        self.assertAlmostEqual(result['per_canonical']['a']['alignment']['correct_mse'],5)
        self.assertAlmostEqual(result['per_canonical']['a']['alignment']['wrong_mse'],6)
        self.assertAlmostEqual(result['per_canonical']['a']['alignment']['improvement_fraction'],1/6)
        self.assertAlmostEqual(result['overall']['alignment']['improvement_fraction'],(1/6+.5)/2)
        self.assertAlmostEqual(result['overall']['variation_ratio'],1.5)
        self.assertAlmostEqual(result['overall']['hf_ratio'],1.5)

    def test_gpu_training_and_inference_do_not_have_cpu_fallback(self):
        with patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CPU training fallback'):
                training.train('/does/not/exist','temporal',0)
        with self.assertRaisesRegex(RuntimeError,'must use GPU'):
            training.load_model('/does/not/exist','temporal',0,device='cpu')

    def test_model_schema_has_no_events_and_shared_initialization_matches(self):
        torch.manual_seed(0);a=training.model_for({'flow_kind':'temporal'})
        torch.manual_seed(0);b=training.model_for({'flow_kind':'pointwise'})
        self.assertEqual(a.cfg.in_dim,86);self.assertEqual(a.cfg.scene_dim,20)
        for name,value in a.state_dict().items():
            if not name.startswith('denoiser.'):
                self.assertTrue(torch.equal(value,b.state_dict()[name]),name)
        config=json.loads(json.dumps(a.config_dict()))
        self.assertEqual(json_digest(config),json_digest(a.config_dict()))
        self.assertFalse(hasattr(a,'head_b'))

    def test_rng_checkpoint_roundtrip_restores_all_non_gpu_streams(self):
        rng=np.random.default_rng(1);torch.manual_seed(2)
        flow=torch.Generator(device='cpu').manual_seed(3)
        with patch.object(torch.cuda,'is_available',return_value=False):state=rng_state(rng)
        state['flow']=flow.get_state()
        buffer=BytesIO();torch.save(state,buffer);buffer.seek(0);saved=torch.load(buffer,weights_only=False)
        expected=(rng.integers(1000,size=8),torch.rand(8),torch.rand(8,generator=flow))
        restore_rng(saved,rng);flow.set_state(saved['flow'])
        actual=(rng.integers(1000,size=8),torch.rand(8),torch.rand(8,generator=flow))
        np.testing.assert_array_equal(expected[0],actual[0]);self.assertTrue(torch.equal(expected[1],actual[1]));self.assertTrue(torch.equal(expected[2],actual[2]))


if __name__=='__main__':unittest.main()
