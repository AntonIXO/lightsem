"""CPU-only tests. Synthetic temporary video lives on the data mount."""
import copy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('optics_media_audit', Path(__file__).resolve().parents[1] / 'scripts/motion/audit_dance_optics_media.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class ExactTimingTests(unittest.TestCase):
    def test_video_exact_30fps(self):
        frames = [{'pts': i*512, 'duration':512} for i in range(6)]
        result = m.timeline(frames, {'time_base':'1/15360'}, 'video')
        self.assertTrue(result['regular_30fps'])
        self.assertEqual(result['decoded_end'], m.q(Fraction(1,5)))
        self.assertEqual(result['decoded_frame_count'], 6)

    def test_video_gap(self):
        frames = [{'pts':0,'duration':1}, {'pts':2,'duration':1}]
        result = m.timeline(frames, {'time_base':'1/30'}, 'video')
        self.assertFalse(result['regular_30fps'])
        self.assertEqual(result['gap_or_overlap_details'][0]['delta'], m.q(Fraction(1,30)))

    def test_audio_variable_samples_exact_end(self):
        frames = [{'pts':0,'nb_samples':1024}, {'pts':1024,'nb_samples':500}]
        result = m.timeline(frames, {'time_base':'1/48000','sample_rate':'48000'}, 'audio')
        self.assertTrue(result['pts_nb_samples_contiguous'])
        self.assertEqual(result['decoded_end'], m.q(Fraction(1524,48000)))
        self.assertEqual(result['sample_count_per_channel'],1524)

    def test_audio_gap_and_overlap(self):
        for pts in (1023,1025):
            result = m.timeline([{'pts':0,'nb_samples':1024}, {'pts':pts,'nb_samples':1024}],
                                {'time_base':'1/48000','sample_rate':'48000'}, 'audio')
            self.assertFalse(result['pts_nb_samples_contiguous'])
            self.assertEqual(result['gap_or_overlap_count'],1)

    def test_missing_pts_fails(self):
        result = m.timeline([{'duration':512}], {'time_base':'1/15360'}, 'video')
        self.assertFalse(result['regular_30fps'])
        self.assertEqual(result['missing_pts_frames'],[0])

    def test_stream_end_integer_not_rounded_decimal(self):
        result = m.stream_end({'time_base':'1/48000','start_pts':-1,'duration_ts':1024,'duration':'0.021'})
        self.assertEqual(result['end'],m.q(Fraction(1023,48000)))

    def test_receipt_no_overwrite_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'receipt.json'; m.write_json(p,{'ok':True})
            self.assertEqual(m.authenticated(p),{'ok':True})
            with self.assertRaises(FileExistsError):m.write_json(p,{'ok':False})
            p.chmod(0o644); p.write_text('{}')
            with self.assertRaises(ValueError):m.authenticated(p)

    def test_artifact_escape(self):
        with self.assertRaises(ValueError):m.safe_child(Path('/tmp/inside'),'../outside')

    def test_incomplete_coverage_never_compares_audio(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); (out/'freeze.json').write_text('{}')
            design={'arms':{str(i):{} for i in range(13)}}
            result=m.coverage(out,{},design)
            self.assertEqual(len(result['members']),15)
            self.assertIsNone(result['audio_comparison'])
            self.assertFalse(result['all_15_audited'])
            self.assertIsNone(result['all_15_exact_media_checks_pass'])


class SyntheticFullDecodeTests(unittest.TestCase):
    def test_software_av1_aac_decode_preserves_counts_and_pcm(self):
        with tempfile.TemporaryDirectory(prefix='optics-media-test-',dir='/mnt/lightsem-data') as d:
            out=Path(d); folder=out/'renders'/'test'; folder.mkdir(parents=True)
            command=['ffmpeg','-v','error','-nostdin','-n','-f','lavfi','-i','color=c=blue:s=640x360:r=30:d=0.2',
                     '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=0.2',
                     '-c:v','libaom-av1','-threads:v','2','-cpu-used','8','-c:a','aac','-threads:a','2',
                     '-t','0.2',str(folder/'video.mp4')]
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            provenance={'test':True}
            with patch.object(m,'render_provenance',return_value=({'generated_frames':8319},provenance)):
                result=m.audit_one(out,'renders','test',{}, {})
            self.assertEqual(result['status'],'complete',result.get('error'))
            self.assertEqual(result['frame_counts']['ffmpeg_actual_decoded'],6)
            self.assertEqual(result['frame_counts']['generated_by_renderer'],8319)
            self.assertFalse(result['checks']['both_decoders_count_8319'])
            self.assertTrue(result['checks']['both_decoders_regular_30fps'])
            self.assertTrue(result['checks']['audio_pts_nb_samples_contiguous'])
            self.assertTrue(result['checks']['pcm_bytes_match_probe_samples'])
            self.assertTrue(result['checks']['all_commands_exit_zero'])
            self.assertTrue(result['checks']['no_error_level_decoder_messages'])
            audit=out/'media_audit/renders__test'
            m.verify_artifacts(audit,m.authenticated(audit/'audit.json')['artifacts'])
            second=out/'pcm_second';second.mkdir()
            pcm=m.pcm_command(result['pcm']['command'],second)
            self.assertEqual(pcm['sha256'],result['pcm']['sha256'])
            self.assertEqual(pcm['bytes'],result['pcm']['bytes'])
            with self.assertRaises(FileExistsError):m.audit_one(out,'renders','test',{}, {})


if __name__=='__main__':unittest.main()
