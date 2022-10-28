import cv2
import numpy as np
import imutils
import math
import time
import serial
import argparse
from collections import deque
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy

# -----------Arduino configuration-----------------
ard = serial.Serial()
ard.port = "COM8"
ard.baudrate = 9600
time.sleep(3)
ard.open()
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video",help="path to the (optional) video file")
#ap.add_argument("-b", "--buffer", type=int, default=50, help="max buffer size")
#args = vars(ap.parse_args())
#font = cv2.FONT_HERSHEY_SIMPLEX

#pts = deque(maxlen=args["buffer"])


def mapObjectPosition(x, y):
    print("[INFO] Object Center coordenates at X0 = {0} and Y0 =  {1}".format(x, y))


alpha_x = 0
alpha_y = 0

def servomotor(cx, cy):  # Arduino function
    if cx > 340:
        if alpha_x > 0:
            if alpha_x > max_v:
                r = 'r' + str(max_v)
                ard.write(str(r).encode())
            else:
                r = 'r' + str(alpha_x)
                ard.write(str(r).encode())
            print('r')
            time.sleep(0.01)

        if alpha_x < 0:
            if abs(alpha_x) > max_v:
                l = 'l' + str(abs(max_v))
                ard.write(str(l).encode())
            else:
                l = 'l' + str(abs(alpha_x))
                ard.write(str(l).encode())
            print('l')
            time.sleep(0.01)

    elif cx < 300:
        if alpha_x > 0:
            if alpha_x > max_v:
                r = 'r' + str(max_v)
                ard.write(str(r).encode())
            else:
                r = 'r' + str(alpha_x)
                ard.write(str(r).encode())
            print('r')
            time.sleep(0.01)
        if alpha_x < 0:
            if abs(alpha_x) > max_v:
                l = 'l' + str(abs(max_v))
                ard.write(str(l).encode())
            else:
                l = 'l' + str(abs(alpha_x))
                ard.write(str(l).encode())
            print('l')
            time.sleep(0.01)
    else:
        ard.write('S'.encode())
        print('S')
        time.sleep(0.01)
    if cy > 260:
        if alpha_y > 0:
            if alpha_y > max_v:
                d = 'd' + str(max_v)
                ard.write(str(d).encode())
            else:
                d = 'd' + str(alpha_y)
                ard.write(str(d).encode())
            print('d')
            time.sleep(0.01)
        if alpha_y < 0:
            if abs(alpha_y) > max_v:
                u = 'u' + str(abs(max_v))
                ard.write(str(u).encode())
            else:
                u = 'u' + str(abs(alpha_y))
                ard.write(str(u).encode())
            print('u')
            time.sleep(0.01)
    elif cy < 220:
        if alpha_y > 0:
            if alpha_y > max_v:
                d = 'd' + str(max_v)
                ard.write(str(d).encode())
            else:
                d = 'd' + str(alpha_y)
                ard.write(str(d).encode())
            print('d')
            time.sleep(0.01)
        if alpha_y < 0 :
            if abs(alpha_y) > max_v:
                u = 'u' + str(abs(max_v))
                ard.write(str(u).encode())
            else:
                u = 'u' + str(abs(alpha_y))
                ard.write(str(u).encode())
            print('u')
            time.sleep(0.01)
    else:
        ard.write('S'.encode())
        print('S')
        time.sleep(0.01)


# --------------------- Camera settings and configuration ---------------------

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# Configuration of the resolution
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Start streaming
profile = pipeline.start(config)

# Create an aligned object
# rs.align allows a performed alignment of depth frames to others frames
# The "align_to" is the stream type to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

'''# to use computer camera instead of real sense
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)'''
# ------------------------------------------------------------------

# time counting
count = 0
inicial = False

# allow the camera to warm up
time.sleep(5.0)

# keep looping
servoPosition = 100
servoPosition1 = 63.5
servoOrientation = 0
errox = []
erroy = []
t = []
t_timestep=[]
t_total=0
iterator=0

while True:
    start_time = time.time()  # to count time to conclude a time step
    count += 1
    if count % 3 != 0:
        continue

    # --------------------- editing frame------------
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    color_image = np.asanyarray(frame.get_data())

    # _, color_image = cap.read() # when using pc camera
    # print(color_image [320,240])

    blurred = cv2.GaussianBlur(color_image, (11, 11), 0)
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([70,155,0])
    upper_blue = np.array([112,255,255])  # [255, 255, 121]

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # center of frame is (320,240)
    (h, w) = color_image.shape[:2]  # w:image-width and h:image-height
    cv2.circle(color_image, (w // 2, h // 2), 7, (255, 0, 255), -1)  # centro da frame em rosa

    cx_frame = w // 2
    cy_frame = h // 2

    # ---------------------------tracking the ball---------------------------------

    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # ('r=' + str(radius))
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            cv2.circle(color_image, (int(x), int(y)), int(radius), (255, 255, 0), 2)
            cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
            # cv2.putText(frame, 'Ball', (int(x) - 50, int(y) + 50), font, 1, (255, 0, 0), 2)
            cv2.putText(color_image, "Bola", (cx - 20, cx - 20), 1, 2.5, (255, 255, 255), 3)
            servomotor(int(x), int(y))

            # initial position of the ball

            if inicial == False:
                cx_i = cx
                cy_i = cy
                inicial = True
                # print( str(cx_i),str(cy_i))

            # coeff = 0.000073  # 1px= 0.000073 meters
            distx = (cx - cx_i)
            disty = (cy - cy_i)

            # compute the error between the center of frame and the center of the ball
            erro_x = (cx - cx_frame)
            erro_y = (cy - cy_frame)
            # print('d_centro_frame_x=' + str(erro_x))
            # print('d_centro_frame_y=' + str(erro_y))

            # compute the velocity of the ball

            tempo = time.time() - start_time
            t_timestep.append(tempo)
            #print ('timestep='+ str(t_timestep))

            # print('tempo=' + str(tempo))
            velocidade_x = distx / tempo
            velocidade_y = disty / tempo
            # print('velocidade_x=' + str(velocidade_x))
            # print('velocidade_y=' + str(velocidade_y))

            # compute number of degrees that servo has to move
            K1 = 0.05  # 0.14
            K2 = 0.007  # 0.1
            K3 = 0.05  # 0.09
            K4 = 0.007  # 0.09
            alpha_x = K1 * erro_x + K2 * velocidade_x
            alpha_x = int(alpha_x)
            print('alpha_x=' + str(alpha_x))
            alpha_y = K3 * erro_y + K4 * velocidade_y
            alpha_y = int(alpha_y)
            print('alpha_y=' + str(alpha_y))

            # setting the max velocity
            # The max velocity of a human neck is 342º/s, which means (352*tempo) will give the max velocity in each time step
            max_v = 342 * tempo
            print(max_v)

            # save the error and time in a list to plot it later
            errox.append(abs(erro_x))
            erroy.append(abs(erro_y))

            t_total = t_total + t_timestep[iterator]
            iterator = iterator + 1 # to count the seconds in all time steps
            t.append(t_total)



            # update variables of time and position
            start_time = time.time()
            cx_i = cx
            cy_i = cy

        else:  # This part aims to find the ball
            ard.write('F'.encode())
            time.sleep(1.00)
            print("F")
            if (servoOrientation == 0):
                if (servoPosition >= 90):
                    servoOrientation = 1
                else:
                    servoOrientation = -1
            if (servoOrientation == 1):
                ard.write('L'.encode())
                time.sleep(1.00)
                servoPosition += 1
                if (servoPosition > 180):
                    servoPosition = 180
                    ard.write('U'.encode())
                    time.sleep(1.00)
                    servoPosition1 += 1
                    if (servoPosition1 > 107):
                        servoPosition1 = 107
                        servoOrientation = -1
            else:
                ard.write('R'.encode())
                time.sleep(1.00)
                servoPosition -= 1
                if (servoPosition < 20):
                    servoPosition = 20
                    ard.write('D'.encode())
                    time.sleep(1.01)
                    servoPosition1 -= 1
                    if (servoPosition1 < 20):
                        servoPosition = 20
                        servoOrientation = 1

    # pts.appendleft(center)
    cv2.imshow("result", color_image)
    # cv2.imshow("mask", mask)
    # cv2.imshow("mask", mask)

    # k = cv2.waitKey(5)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

pipeline.stop()
# cap.release
cv2.destroyAllWindows()


# plot the distance error to compare the influence of using different K's

plt.plot(t, errox, label='Eixo do X')
plt.plot(t, erroy, label='Eixo do Y')
max_x = max(errox)
max_y = max(erroy)
tmax_y = t[np.argmax(erroy)]
tmax_x = t[np.argmax(errox)]
plt.plot(tmax_x, max_x, 'r*', label= 'máximo em x = '+str(max_x))
plt.plot(tmax_y, max_y, 'r*', label='máximo em y = '+str (max_y))


plt.ylabel('Distância em píxeis')
plt.xlabel('t/s')
plt.title('Variação da distância do centro do objeto ao centro da imagem ')
plt.legend()
plt.show()
