"""Eye contact preview tool."""
import argparse
import cv2
import numpy as np


def main(sample, video, outp, vertical):
    """Main function."""
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
        ret, frame = cap.read()
        if ret is not True:
            break

        axis = 0 if vertical else 1
        frame = np.concatenate((frame, frame), axis=axis)
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
