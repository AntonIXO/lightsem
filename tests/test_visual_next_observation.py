from dataclasses import fields
import numpy as np
from lightsem.experiments.visual_next.observation import FrameObservation
from lightsem.experiments.visual_next.features import spatial_color_grid
from lightsem.model.light_layers.targets import extract_color_targets


def test_reused_hsv_is_exact_for_all_original_target_fields_and_grid():
    rng=np.random.default_rng(47)
    images=[rng.integers(0,256,(25,31,3),dtype=np.uint8),np.zeros((24,32,3),np.uint8),
            np.full((24,32,3),128,np.uint8),np.tile(np.array([255,0,1],np.uint8),(24,32,1))]
    for rgb in images:
        observer=FrameObservation(rgb)
        for mask in (np.ones(rgb.shape[:2],bool),np.zeros(rgb.shape[:2],bool),rng.random(rgb.shape[:2])>.3):
            original=extract_color_targets(rgb,mask);cached=observer.target(mask)
            for field in fields(original):
                a,b=getattr(original,field.name),getattr(cached,field.name)
                if isinstance(a,np.ndarray):assert np.array_equal(a,b),field.name
                else:assert a==b,field.name
            assert np.array_equal(observer.grid(mask),spatial_color_grid(rgb,mask))
