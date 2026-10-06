"""Reuse fixed diagnostics with explicit D4-max scientific comparators."""
from pathlib import Path
import importlib.util
import numpy as np
from ..dynamic_dance_combo import diagnostics as common
from ..dynamic_dance_combo.campaign import sha


def source_response(candidate,other,reference=None):
 original=candidate if reference is None else reference
 result=common._difference(original,other)
 a,b=original['body'].metadata,other['body'].metadata
 result['fixed_inputs']={'G78_library_sha256':a.get('generated_source_sha256'),'same_G78_library':a.get('generated_source_sha256')==b.get('generated_source_sha256'),'bank_sha256':a.get('bank_sha256'),'same_bank':a.get('bank_sha256')==b.get('bank_sha256'),'analyzed_audio_changed':a.get('music_timeline_sha256')!=b.get('music_timeline_sha256')}
 result['scope']='Audio query controls for composition and pacing; saved G78 library and original tags fixed. This does not measure fresh D7 generation sensitivity.'
 result['global_brightness_envelope_held_fixed']=bool(np.array_equal(original['arrays']['brightness_envelope'],other['arrays']['brightness_envelope']))
 return result


def compare(root,method,song):
 # Private module identity prevents changing other campaigns in this process.
 spec=importlib.util.spec_from_file_location('_d4max_fixed_diagnostics',common.__file__)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.COMPARATORS={'C3':'S4','C4':'C3'}
 module.CONTROLS={'C3':('C3_shuffled','D7_reference'),'C4':('C4_loudness','C4_shifted','C4_unanchored','C4_zero_allowed')}
 module._source_response=source_response
 original_write=module._write
 def write(path,value):
  if 'primary' in value:
   value['primary']['diagnostic_id']='dd-d4max-d8-emitted-'+('recurrence' if method=='C3' else 'release')+'-lift'
   value['inputs'][str(Path(__file__))]=sha(__file__)
  return original_write(path,value)
 module._write=write
 return module.compare(root,method,song)
