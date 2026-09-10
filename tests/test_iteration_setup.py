from pathlib import Path
import tempfile
import unittest
from lightsem.experiments.iteration_speed import engineering_inputs


class SetupTests(unittest.TestCase):
    def test_clean_screen_does_not_require_or_create_repair_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertEqual(engineering_inputs(root),[root/'test_report.json',root/'engineering.json'])
            self.assertEqual(list(root.iterdir()),[])

    def test_actual_incident_archive_is_preserved_in_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'engineering_before_compiler_repair.json'
            archive.write_text('{"incident":"retained"}\n')
            self.assertEqual(engineering_inputs(root),[root/'test_report.json',root/'engineering.json',archive])
            self.assertEqual(archive.read_text(),'{"incident":"retained"}\n')


if __name__=='__main__':unittest.main()
