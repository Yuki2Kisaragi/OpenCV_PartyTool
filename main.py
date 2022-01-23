import cv2
import sys
from datetime import datetime

# 0:内蔵カメラ使用
BUILT_IN_CAMERA = 0

# Cascade File PATH
Face_Cascade_FILE_PATH = "./HaarCascades/haarcascade_frontalface_default.xml"
Eye_Cascade_FILE_PATH = "./HaarCascades/haarcascade_eye.xml"

# Mode
MODE_NORMAL = 0
MODE_DETECTION_FACE_AND_EYE = 1
MODE_DETECTION_FACE = 2


class ImageOpenError(Exception):
    pass


class DeviceCanNotFoundError(Exception):
    pass


def main():
    """
    main

    Raises:
        DeviceCanNotFoundError: [description]
        ImageOpenError: [description]
    """
    face_cascade = cv2.CascadeClassifier(Face_Cascade_FILE_PATH)
    eye_cascade = cv2.CascadeClassifier(Eye_Cascade_FILE_PATH)

    mode = MODE_NORMAL

    try:

        capture = cv2.VideoCapture(BUILT_IN_CAMERA)

        if capture.isOpened():
            print("Cammera is found")

            while True:
                ret, frame = capture.read()

                if ret is False:
                    # Cammera cannot found
                    raise DeviceCanNotFoundError

                if mode == MODE_DETECTION_FACE_AND_EYE:
                    frame = detect_face_and_eye(face_cascade, eye_cascade, frame)
                elif mode == MODE_DETECTION_FACE:
                    frame = detect_face(face_cascade, frame)

                cv2.imshow("capture", frame)

                input_key = cv2.waitKey(1)

                if input_key == ord('q') or input_key == 27:
                    # Capture is Terminated.
                    cv2.destroyWindow("capture")
                    break
                elif input_key == ord('c'):
                    # cv2.imwrite("capture.jpg", frame)
                    save_image(frame)
                    continue
                elif input_key == ord('f'):
                    if mode != MODE_DETECTION_FACE_AND_EYE:
                        mode = MODE_DETECTION_FACE_AND_EYE
                    else:
                        mode = MODE_NORMAL
                    continue
                elif input_key == ord('e'):
                    if mode != MODE_DETECTION_FACE:
                        mode = MODE_DETECTION_FACE
                    else:
                        mode = MODE_NORMAL
                    continue

        else:
            raise ImageOpenError

    except ImageOpenError as e:
        print(e)
        sys.exit()

    except DeviceCanNotFoundError as e:
        print(e)
        sys.exit()

    finally:
        print("End")


def save_image(img):
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = "Output/capture_" + date + ".png"
    cv2.imwrite(path, img)  # ファイル保存
    print(f"Image File is saved as '{path}'.")


def detect_face_and_eye(face_cascade, eye_cascade, frame):
    """detection face and eyes from video capture

    Args:
        face_cascade ([type]): [description]
        eye_cascade ([type]): [description]
        frame ([type]): [description]

    Returns:
        [type]: [description]
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame_gray)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = frame[y: y + h, x: x + w]
        face_gray = frame_gray[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return frame


def detect_face(face_cascade, frame):
    """detect_face

    Args:
        face_cascade ([type]): [description]
        frame ([type]): [description]

    Returns:
        [type]: [description]
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame_gray)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame


def detect_face_and_eye_from_picture(input_img: str, saveFileName: str) -> None:

    face_cascade = cv2.CascadeClassifier(Face_Cascade_FILE_PATH)
    eye_cascade = cv2.CascadeClassifier(Eye_Cascade_FILE_PATH)

    src = cv2.imread(input_img)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(src_gray)

    for x, y, w, h in faces:
        cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = src[y: y + h, x: x + w]
        face_gray = src_gray[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imwrite(saveFileName, src)


if __name__ == "__main__":
    main()
