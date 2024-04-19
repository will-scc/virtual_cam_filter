import cv2
import time
import pyvirtualcam

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

time.sleep(2)

# -----------------------------------------------
# Face Detection using DNN Net
# -----------------------------------------------
# detect faces using a DNN model 
# download model and prototxt from https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison/models

def detectFaceOpenCVDnn(net, frame, conf_threshold=0.5):
    
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)0
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8,)
            
            top=x1
            right=y1
            bottom=x2-x1
            left=y2-y1

            #  blurry rectangle to the detected face
            face = frame[right:right+left, top:top+bottom]
            face = cv2.blur(face,(50,50), 35)
            frame[right:right+face.shape[0], top:top+face.shape[1]] = face

    return frame, bboxes

# load face detection model
modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

detectionEnabled = True

with pyvirtualcam.Camera(width=1280, height=720, fps=60) as cam:
    while True:
        try:
            _, frame = video_capture.read()

            if(detectionEnabled == True):
                outOpencvDnn, bboxes = detectFaceOpenCVDnn(net, frame)

            cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
            
            #cam.send(frame)
            #cam.sleep_until_next_frame()
            # Increase brightness
            #brightness_factor = 2.5  # Adjust this value as needed
            #brightened_frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)

            cv2.imshow('Face Blur using DNN', frame)

        except Exception as e:
            print(f'exc: {e}')
            pass

        # key controllerq
        key = cv2.waitKey(1) & 0xFF    
        if key == ord("d"):
            detectionEnabled = not detectionEnabled

        if key == ord("q"):
            break

video_capture.release()
cv2.destroyAllWindows()