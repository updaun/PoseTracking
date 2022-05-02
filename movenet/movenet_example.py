import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time

# Load Model
model_path = "movenet/models/lite-model_movenet_singlepose_lightning_3.tflite"
model_data_type = tf.float32
# model_path = "movenet/models/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite"
# model_data_type = tf.uint8
# model_path = "movenet/models/lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite"
# model_data_type = tf.uint8
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

# Make Detections
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# initialize value time for fps
pTime = 0
cTime = 0

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)


while cap.isOpened():
    ret, frame = cap.read()

    h, w, c = frame.shape

    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    #plt.imshow(tf.cast(np.squeeze(img), dtype=tf.int32))
    input_image = tf.cast(img, dtype=model_data_type)

    # Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make prediction
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    # print(keypoints_with_scores[0][0])

    # Get landmark list
    lmList = []
    for index, lm in enumerate(keypoints_with_scores[0][0]):
        lmList.append([index, int(lm[1]*w), int(lm[0]*h)])
    print(lmList[0])

    # Rendering
    draw_connections(frame, keypoints_with_scores, KEYPOINT_EDGE_INDS_TO_COLOR, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)

    # FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)



    cv2.imshow("MoveNet Lighting", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()