from pathlib import Path
import tempfile
import unittest
from lightsem.experiments.iteration_speed import engineering_inputs
from lightsem.experiments.training_runtime import TrainingRuntime
from lightsem.experiments.night_freeze import training_config


class SetupTests(unittest.TestCase):
    def test_new_defaults_reach_frozen_training_config(self):
        runtime=TrainingRuntime()
        expected={'precision':'bf16','optimizer':'adamw','compile':'default','prefetch':2}
        self.assertEqual(runtime.as_dict(),expected)
        cfg=training_config(15200,{'precision':runtime.precision,'runtime':runtime.as_dict()})
        self.assertEqual(cfg['batch'],32)
        self.assertEqual(cfg['precision'],'bf16')
        self.assertEqual(TrainingRuntime.from_training(cfg).as_dict(),expected)

    def test_old_protocols_do_not_inherit_new_defaults(self):
        legacy=TrainingRuntime.legacy().as_dict()
        self.assertEqual(TrainingRuntime.from_training({}).as_dict(),legacy)
        self.assertEqual(TrainingRuntime.from_training({'precision':'fp32'}).as_dict(),legacy)
        self.assertEqual(TrainingRuntime.from_training({'runtime':{'precision':'bf16'}}).as_dict(),
                         TrainingRuntime.legacy('bf16').as_dict())
        with self.assertRaises(ValueError):
            TrainingRuntime.from_training({'precision':'fp32','runtime':TrainingRuntime().as_dict()})

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
