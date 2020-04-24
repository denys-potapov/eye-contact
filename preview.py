"""Eye contact preview tool."""
import argparse
import cv2
import numpy as np
from eye_contact import EyeContact


def main(sample, video, outp, vertical):
    """Main function."""
    open_img = cv2.imread(sample)
    eye_contact = EyeContact(open_img)

    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if vertical:
        h = h * 2
    else:
        w = w * 2

    out = cv2.VideoWriter(
        outp,
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        fps, (w, h))

    while(cap.isOpened()):
        ret, orig = cap.read()
        if ret is not True:
            break
        new = eye_contact.open_eyes(orig)

        axis = 0 if vertical else 1
        frame = np.concatenate((orig, new), axis=axis)

        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = 10, 50
        cv2.putText(
            frame, 'Original', (x, y), font,
            1, (255, 0, 0), 2, cv2.LINE_AA)
        if vertical:
            y = y + h // 2
        else:
            x = x + w // 2
        cv2.putText(
            frame, 'Eye contact', (x, y), font,
            1, (255, 0, 0), 2, cv2.LINE_AA)
        out.write(frame)

    out.release()
    cap.release()


if __name__ == '__main__':
    desc = """Stand alone eye contact previwe

    Sample usage:
        python3 preview.py open.jpg video.mp4 out.avi --vertical
    """

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=desc)

    p.add_argument('sample', type=str, help='open eye frame')
    p.add_argument('video', type=str, help='logo image')
    p.add_argument('outp', type=str, help='output video file')
    p.add_argument(
        '--vertical', help="stack results vertiacl", action="store_true")
    args = p.parse_args()

    result = main(args.sample, args.video, args.outp, args.vertical)
