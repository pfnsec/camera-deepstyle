
import time
import cv2
import numpy as np
import pyvirtualcam

from evaluate_cur import sess, img_placeholder, preds

verbose = False

# Set up webcam capture
vc = cv2.VideoCapture(0) # 0 = default camera

if not vc.isOpened():
    raise RuntimeError('could not open video source')

pref_width = 1280
pref_height = 720
pref_fps_in = 30
vc.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
vc.set(cv2.CAP_PROP_FPS, pref_fps_in)

# Query final capture device values (may be different from preferred settings)
width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = vc.get(cv2.CAP_PROP_FPS)
print(f'webcam capture started ({width}x{height} @ {fps_in}fps)')

fps_out = 20

try:
    delay = 0 # low-latency, reduces internal queue size

    with pyvirtualcam.Camera(width, height, fps_out, delay, print_fps=True) as cam:
        print(f'virtual cam started ({width}x{height} @ {fps_out}fps)')

        while True:
            # Read frame from webcam
            rval, in_frame = vc.read()
            if not rval:
                raise RuntimeError('error fetching frame')

            in_frame = cv2.resize(in_frame, (1280, 692))

            in_frame.shape = (1, 692, 1280, 3)

            _preds = sess.run(preds, feed_dict={img_placeholder: in_frame})

            out_frame = _preds[0, :, :, :]

            # convert to RGBA
            out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
            out_frame = cv2.resize(out_frame, (1280, 720))
            out_frame_rgba = np.zeros((height, width, 4), np.uint8)
            out_frame_rgba[:,:,:3] = out_frame
            out_frame_rgba[:,:,3] = 255

            # Send to virtual cam
            cam.send(out_frame_rgba)

            # Wait until it's time for the next frame
            cam.sleep_until_next_frame()
finally:
    vc.release()