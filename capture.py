import cv2
import sys


class ImageOpenError(Exception):
    pass


class DeviceCanNotFoundError(Exception):
    pass


def get_capture_properties(capture: cv2.VideoCapture) -> tuple:
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"frame size = {width} x {height}")
    return (width, height)


# 0:内蔵カメラ使用
BUILT_IN_CAMERA = 0


def main():
    try:
        capture = cv2.VideoCapture(BUILT_IN_CAMERA)

        if capture.isOpened():
            print("Cammera is found")

            while True:
                ret, frame = capture.read()
                if ret is False:
                    # Cammera cannot found
                    raise DeviceCanNotFoundError

                cv2.imshow("capture", frame)

                input_key = cv2.waitKey(1)

                if input_key == ord('q') or input_key == 27:
                    # Capture is Terminated.
                    cv2.destroyWindow("capture")
                    break
                elif input_key == ord('c'):
                    cv2.imwrite("capture.jpg", frame)
                    print("Image File is saved as `capture.jpg` .")
                    continue
                elif input_key == ord('f'):
                    pass

        else:
            raise ImageOpenError

    except ImageOpenError as e:
        print(e)
        sys.exit()

    except DeviceCanNotFoundError as e:
        print(e)
        sys.exit()

    except:
        print("Error", sys.exc_info()[0])
        print(sys.exc_info()[1])
        import traceback
        print(traceback.format_tb(sys.exc_info()[2]))

    finally:
        print("End")


if __name__ == "__main__":
    main()
