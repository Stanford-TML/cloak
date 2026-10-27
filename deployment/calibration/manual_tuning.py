"""Required recorded-frame overlay review. No robot connections or movement."""

import numpy as np

from deployment.calibration.hand_eye import transform, pose_vector


def adjust(pose, axis, amount):
    """Apply a local camera-frame translation or rotation."""
    delta = np.zeros(6)
    delta[axis] = amount
    return pose_vector(transform(pose) @ transform(delta))


def tune(initial_pose, frames, joints, grippers, intrinsics, embodiment):
    import tkinter as tk
    from tkinter import messagebox
    import cv2
    from PIL import Image, ImageTk
    from openpi.calibration.silhouette import Sim

    sim = Sim(embodiment)
    root = tk.Tk()
    root.title(f"Cloak calibration: {embodiment}")
    pose = np.asarray(initial_pose).copy()
    index, fine, accepted = 0, True, False
    visited = set()
    visible = False
    canvas = tk.Label(root)
    canvas.pack()
    status = tk.StringVar()
    tk.Label(root, textvariable=status, justify="left").pack()
    tk.Label(
        root,
        text="Arrows: X/Y | PageUp/PageDown: Z | Q/A W/S E/D: rotate X/Y/Z\n"
        "F: fine/coarse | N/P: next/previous frame | R: reset | Enter: accept | Esc: cancel",
    ).pack()

    def draw():
        nonlocal visible
        visited.add(index)
        ee = sim.set_pose(joints[index], float(grippers[index]))
        k = np.asarray(intrinsics).copy()
        h, w = frames[index].shape[:2]
        k[0] *= 320 / w
        k[1] *= 180 / h
        mask = sim.render_gripper_mask(ee @ transform(pose), float(k[1, 1]))
        # Match deployment exactly: its renderer uses fy with a centered,
        # square-pixel camera. Manual refinement compensates the same model.
        visible = bool(mask.any())
        rgb = cv2.resize(frames[index], (320, 180))
        rgb[mask] = (0.5 * rgb[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        picture = ImageTk.PhotoImage(Image.fromarray(rgb).resize((960, 540)))
        canvas.configure(image=picture)
        canvas.image = picture
        status.set(
            f"Frame {index + 1}/{len(frames)} | {'Fine: 1 mm / 0.5 deg' if fine else 'Coarse: 5 mm / 5 deg'}\n"
            f"xyz (m): {np.round(pose[:3], 5)} | xyz (deg): {np.round(np.rad2deg(pose[3:]), 2)}"
        )

    def key(event):
        nonlocal pose, index, fine, accepted
        name = event.keysym.lower()
        mapping = {
            "left": (0, -1),
            "right": (0, 1),
            "up": (1, 1),
            "down": (1, -1),
            "prior": (2, 1),
            "next": (2, -1),
            "q": (3, 1),
            "a": (3, -1),
            "w": (4, 1),
            "s": (4, -1),
            "e": (5, 1),
            "d": (5, -1),
        }
        if name in mapping:
            axis, sign = mapping[name]
            step = (0.001 if fine else 0.005) if axis < 3 else np.deg2rad(0.5 if fine else 5)
            pose = adjust(pose, axis, sign * step)
        elif name == "f":
            fine = not fine
        elif name in ("n", "p"):
            index = (index + (1 if name == "n" else -1)) % len(frames)
        elif name == "r":
            pose = np.asarray(initial_pose).copy()
        elif name == "escape":
            root.destroy()
            return
        elif name == "return":
            if not visible:
                messagebox.showinfo(
                    "End effector not visible", "Adjust the camera until the rendered end effector is visible."
                )
                return
            if len(visited) < min(3, len(frames)):
                messagebox.showinfo("Review more frames", "Inspect at least three frames using N/P before accepting.")
                return
            accepted = messagebox.askyesno("Save calibration", "Accept this alignment and save it for deployment?")
            if accepted:
                root.destroy()
                return
        draw()

    root.bind("<Key>", key)
    try:
        draw()
        root.mainloop()
    finally:
        sim.renderer.close()
    return pose if accepted else None
