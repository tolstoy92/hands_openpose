import PyOpenPose as OP
import time
import cv2

import numpy as np
import os

#   WHY matplotlib?!
#from matplotlib import pyplot as plt

OPENPOSE_ROOT = "/home/user/openpose"

def ComputeBB(hand, padding=1.5):
    minX = np.min(hand[:, 0])
    minY = np.min(hand[:, 1])

    maxX = np.max(hand[:, 0])
    maxY = np.max(hand[:, 1])

    width = maxX - minX
    height = maxY - minY

    cx = minX + width/2
    cy = minY + height/2

    width = height = max(width, height)
    width = height = width * padding

    minX = cx - width/2
    minY = cy - height/2

    score = np.mean(hand[:, 2])

    return score, [int(minX), int(minY), int(width), int(height)]


def run(cap):
    ret, frame = cap.read()         # Get frames from camera

    imgSize = frame.shape[1::-1]    # Get Img size

    # Flags to OP.OpenPose

    download_heatmaps = False
    with_face = False

    with_hands = True

    op = OP.OpenPose((656, 368), (240, 240), tuple(imgSize), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                     download_heatmaps, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)

    actual_fps = 0
    paused = False
    delay = {True: 0, False: 1}
    RUN = True

    newHandBB_R = initHandBB_R = handBB_R = [220, 190, 200, 200]
    newHandBB_L = initHandBB_L = handBB_L = [50, 190, 200, 200]

    while RUN:
        ret, frame = cap.read()
        start_time = time.time()

        box_R = np.array(handBB_R + [0, 0, 0, 0], dtype=np.int32)       # Box for right hand. Zeros?!
        box_L = np.array(handBB_L + [0, 0, 0, 0], dtype=np.int32)       # Box for left hand. Zeros?!

        op.detectHands(frame, box_R.reshape((1, 8)))        # Detect hand in right box.
        op.detectHands(frame, box_L.reshape((1, 8)))        # Detect hand in left box.

        t = time.time() - start_time
        op_fps = 1.0 / t

        res = op.render(frame)

        cv2.putText(res, 'UI FPS = %f, OP-HAND FPS = %f. Press \'r\' to reset.' % (actual_fps, op_fps), (20, 20), 0,
                    0.5, (0, 0, 255))

        cv2.rectangle(res, (handBB_R[0], handBB_R[1]), (handBB_R[0] + handBB_R[2], handBB_R[1] + handBB_R[3]), [50, 155, 50], 2)
        cv2.rectangle(res, (newHandBB_R[0], newHandBB_R[1]), (newHandBB_R[0] + newHandBB_R[2], newHandBB_R[1] + newHandBB_R[3]),
                      [250, 55, 50], 1)

        cv2.rectangle(res, (handBB_L[0], handBB_L[1]), (handBB_L[0] + handBB_L[2], handBB_L[1] + handBB_L[3]),
                      [50, 155, 50], 2)
        cv2.rectangle(res, (newHandBB_L[0], newHandBB_L[1]),
                      (newHandBB_L[0] + newHandBB_L[2], newHandBB_L[1] + newHandBB_L[3]),
                      [250, 55, 50], 1)

        cv2.imshow("OpenPose result", res)

        leftHand = op.getKeypoints(op.KeypointType.HAND)[0].reshape(-1, 3)
        rightHand = op.getKeypoints(op.KeypointType.HAND)[0].reshape(-1, 3)


        score_L, newHandBB_L = ComputeBB(leftHand)
        score_R, newHandBB_R = ComputeBB(rightHand)


        if score_L > 0.5 and score_R > 0.5:  # update BB only when score is good.
            handBB_L = newHandBB_L
            handBB_R = newHandBB_R

        if cv2.waitKey(10) & 0xFF == 27:
            RUN = not RUN

        # key = cv2.waitKey(delay[paused])
        # if key & 255 == ord('p'):
        #     paused = not paused
        #
        # if key & 255 == ord('q'):
        #     cap.release()
        #     RUN = not RUN
        #
        # if key & 255 == ord('r'):
        #     handBB_R = initHandBB_R
        #     handBB_L = initHandBB_L

        actual_fps = 1.0 / (time.time() - start_time)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    run(cap)