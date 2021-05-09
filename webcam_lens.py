
import time
import cv2
import numpy as np
import pyvirtualcam

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

            # Apply cartoon filter
            # (https://pysource.com/2018/10/11/how-to-create-a-cartoon-effect-opencv-with-python/)
            # 1) Edges
            #gray = cv2.cvtColor(in_frame, cv2.COLOR_BGR2GRAY)
            #gray = cv2.medianBlur(gray, 5)
            #edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            # 2) Color
            #color = cv2.bilateralFilter(in_frame, 9, 300, 300)
            # 3) Cartoon
            #cartoon = cv2.bitwise_and(color, color, mask=edges)
            #assert cartoon.shape == (height, width, 3)
            #in_frame = cv2.resize(in_frame, (1280, 692))

            distCoeff = np.zeros((4,1),np.float64)

            # TODO: add your coefficients here!
            k1 = -1.0e-4 # negative to remove barrel distortion
            k2 = 0.00
            p1 = 0.0
            p2 = 0.0

            distCoeff[0,0] = k1
            distCoeff[1,0] = k2
            distCoeff[2,0] = p1
            distCoeff[3,0] = p2

            # assume unit matrix for camera
            mat = np.eye(3,dtype=np.float32)

            mat[0,2] = width/2.0  # define center x
            mat[1,2] = height/2.0 # define center y
            mat[0,0] = 10.        # define focal length x
            mat[1,1] = 10.        # define focal length y

            # here the undistortion will be computed
            out_frame = cv2.undistort(in_frame, mat, distCoeff)


            # convert to RGBA
            out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
            #out_frame = cv2.resize(out_frame, (1280, 720))

            out_frame_rgba = np.zeros((height, width, 4), np.uint8)
            out_frame_rgba[:,:,:3] = out_frame
            out_frame_rgba[:,:,3] = 255

            # Send to virtual cam
            cam.send(out_frame_rgba)

            # Wait until it's time for the next frame
            cam.sleep_until_next_frame()
finally:
    vc.release()