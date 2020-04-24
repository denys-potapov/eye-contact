"""Photo capture."""
import cv2
import argparse


if __name__ == '__main__':
    desc = """Capture photo

    Sample usage:
        python3 capture.py open.jpg /dev/video0
    """

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=desc)

    p.add_argument('dest', type=str, help='photo output')
    p.add_argument('video', type=str, help='video device ex. /dev/vidoe0')
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) != -1:
            break

    cv2.imwrite(args.dest, frame)

    cap.release()
    cv2.destroyAllWindows()
