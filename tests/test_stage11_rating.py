import json
from pathlib import Path
import tempfile
import unittest
from lightsem.experiments.stage11_rating import rate
from lightsem.experiments.stage11_common import digest

class RatingTests(unittest.TestCase):
    def test_unknown_and_tie_are_distinct_and_resume_is_per_rater(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'results/renders';folder.mkdir(parents=True)
            for side in ('A','B'):(folder/f'p_{side}.mp4').write_bytes(side.encode())
            pair={'id':'p','videos':{s:f'p_{s}.mp4' for s in ('A','B')},
                  'video_sha256':{s:digest(folder/f'p_{s}.mp4') for s in ('A','B')}}
            (folder/'pairs.json').write_text(json.dumps({'pairs':[pair]}))
            responses=iter(['u','=']);played=[]
            self.assertTrue(rate(root,'anton',input_fn=lambda _:next(responses),player=played.append))
            first=(folder/'ratings.jsonl').read_text();record=json.loads(first)
            self.assertEqual(record['fits_music'],'u');self.assertEqual(record['engineer'],'=')
            self.assertTrue(rate(root,'anton',input_fn=lambda _:self.fail('must resume'),player=played.append))
            self.assertEqual((folder/'ratings.jsonl').read_text(),first)
            self.assertEqual(len(played),2)
            responses=iter(['a','q'])
            self.assertFalse(rate(root,'second',input_fn=lambda _:next(responses),player=played.append))
            self.assertEqual((folder/'ratings.jsonl').read_text(),first)

if __name__=='__main__':unittest.main()
