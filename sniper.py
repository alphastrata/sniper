"""
SNIPER
(҂`_´) ︻╤╦╤─ ҉- - - - - -
A simple script to take stills from RTSP feeds when a person is in frame.
Made to cutdown the ammount of time I spent going through camera footage from outfront the house.
"""

import cv2
import datetime
import mediapipe as mp
from datetime import datetime
import logging
import sys
from multiprocessing import Process


def parseArgs():
    """When snipe.py is run, it should be doneso with a camera that you want to collect stills from.
    Some convenient names are provided in in parseCameraAddress"""
    arg = sys.argv[1]
    if arg and len(sys.argv) > 1:
        return arg
    log("NO_CAMERA", "Incorrect arguments. See terminal output for help")
    print(
        "You need to pass a camera, options are:\nfront, garage.\n\nexample: `python3 sniper.py breakroom`"
    )

CAMERA = parseArgs()


def parseCameraAddress(CAMERA=CAMERA):
    """Camera RTSP addresses are tedious to type as args, so to make it easier for end users to type less we're reducing their options."""
    match CAMERA:
        case "front":
            return  "rtsp://<username>:<password>@<ipaddress>"  
        case "garage":
            return  "rtsp://<username>:<password>@<ipaddress>"  
        case _:
            print("You need to pass a camera, options are:\nbreakroom, front, garage.\n\nexample: `python3 sniper.py breakroom`")


def log(LOGTYPE=CAMERA, msg=None):
    """A simple logger. Located YYYY-MM-DD.log"""
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    logging.basicConfig(
        filename=f"{today}.log", level=logging.INFO, format="%(asctime)s %(message)s"
    )
    logging.info(f" __ {LOGTYPE.upper()}, {msg}")


def savePng(filename, image):
    """a wrapper around imwrite from the openCV library to make it easier to use with the multiprocessing library."""
    cv2.imwrite(filename, image)
    print("imwrite just ran for {filename}")


def run(capID, CAMERA=CAMERA, display_for_debug=True):
    """Detect and humans and save stills when detected at stills/CAMERA_TIMESTAMP.png
    This implementation uses the mediapipe library's models to do so."""
    mp_face_detection = mp.solutions.face_detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(capID)
    
    workers = [] # a place to store thread join handles

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=4.6
    ) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                p = Process(target=log, args=(CAMERA, "FAILED TO ACQUIRE IMAGE."))
                p.start()
                workers.append(p)
                continue
            image.flags.writeable = False
            results = face_detection.process(image)
            if results.detections:
                now = datetime.now()
                ts = now.strftime("%Y-%m-%d %H-%M-%S")
                filename = f"{CAMERA}_{ts}.png"
                ## Write to log and save files on their own threads
                l = Process(target=log, args=(CAMERA, "DETECTED A PERSON"))
                s = Process(target=savePng, args=(filename, image))
                l.start()
                workers.append(l)
                s.start()
                workers.append(s)
                
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)

#============== DEBUG: show the camera feed============== 
                if display_for_debug:
                    cv2.imshow(f"{CAMERA}", image)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                    cv2.waitKey(1)
#========================================================= 

            ## Join the threads 
            for worker in workers:
                worker.join()
                log("WORKERTHREAD", "CLOSED")

    cap.release()


if __name__ == "__main__":
    camera_id = parseCameraAddress(CAMERA)
    run(camera_id, CAMERA)
    cv2.destroyAllWindows()

