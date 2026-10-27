"""Diagnostic frames keep each inference mask paired with its own input image."""
import cv2
import numpy as np

def diagnostic_frame(external, wrist, inference_wrist=None, mask=None):
    def panel(img, label):
        out = cv2.resize(img, (640, 360)).copy()
        cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
        return out
    if mask is None or inference_wrist is None:
        masked = np.zeros((180,320,3), dtype=np.uint8)
        binary = masked.copy()
        label = "Mask unavailable"
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != inference_wrist.shape[:2]:
            raise ValueError("Inference mask and input image must have the same resolution")
        masked = inference_wrist.copy()
        masked[mask] = (.45 * masked[mask] + .55 * np.array([0,255,0])).astype(np.uint8)
        binary = np.repeat((mask.astype(np.uint8)*255)[...,None],3,axis=2)
        label = "Last inference input + source mask"
    return np.concatenate([
        np.concatenate([panel(external,"External"),panel(wrist,"Wrist")],axis=1),
        np.concatenate([panel(masked,label),panel(binary,"Source pixel mask (before patch masking)")],axis=1)
    ],axis=0)
