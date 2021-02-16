from pyk4a import Config, PyK4A
import numpy as np
from typing import Tuple
import pyk4a

def read_frame() -> Tuple[bool,np.ndarray]:
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            color_format=pyk4a.ImageFormat.COLOR_BGRA32,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=False,

        )
    )

    k4a.start()
    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510
    frame = k4a.get_capture().depth
    if np.any(frame):
        frame = np.clip(frame, 0, 2**10 - 1)
        frame >>= 2
        return True, frame.astype(np.uint8)
    return False, None
