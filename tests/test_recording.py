import numpy as np
import pytest
from deployment.recording import diagnostic_frame

def test_mask_overlay_uses_matching_inference_frame():
    current = np.full((180,320,3), 200, np.uint8)
    inference = np.zeros_like(current)
    mask = np.zeros((180,320), bool); mask[70:110,120:200] = True
    out = diagnostic_frame(current,current,inference,mask)
    assert out.shape == (720,1280,3)
    assert out[540,320,1] > 0 and out[540,320,0] == 0
    assert np.all(out[540,960] == 255)
    assert np.all(out[300,320] == 200)
    assert not inference.any()

def test_missing_mask_and_shape_validation():
    img=np.zeros((180,320,3),np.uint8)
    assert diagnostic_frame(img,img).shape == (720,1280,3)
    with pytest.raises(ValueError):
        diagnostic_frame(img,img,img,np.zeros((5,5)))
