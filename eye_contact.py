"""Fake eye contact."""
import argparse
import cv2
import dlib
import fcntl
import math
import numpy as np
import lib.v4l2 as v4l2
from threading import Thread
from time import perf_counter

MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'

RIGHT_EYE_RANGE = (36, 42)
LEFT_EYE_RANGE = (42, 48)

EYE_SCALE = 1.5
EYE_BLUR = 21

DEFAULT_SCALE = 320

SHOW_FPS_EVEVRY = 10


def centers2(points):
    """Return eye centers and averall center."""
    centers = [p.mean(axis=0) for p in points]
    center = np.array(centers).mean(axis=0)
    return center, centers


def scale_box2d(box, scale):
    """Scale box."""
    h, w = box[1]
    return (box[0], (h * scale, w * scale), box[2])


def to_polar(p1, p2):
    """Get a length and angel in degrees."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))


class EyeContact:
    """EyeContact."""

    def __init__(self, img, scale=DEFAULT_SCALE):
        """Init the eyes replacer."""
        self.img = img
        self.mask = np.zeros(img.shape[:2], dtype=np.uint8)
        self.scale = scale
        # dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(MODEL_PATH)

        elipses = []
        eyes = self.detect_eyes(img)
        for eye in eyes:
            ellipse = cv2.fitEllipse(eye)
            mask_ellipse = scale_box2d(ellipse, EYE_SCALE)
            elipses.append(mask_ellipse)
            cv2.ellipse(self.mask, mask_ellipse, 255, -1)

        self.mask = cv2.GaussianBlur(self.mask, (EYE_BLUR, EYE_BLUR), 0)

        # crop everything
        points = list(
            cv2.boxPoints(elipses[0])) + list(cv2.boxPoints(elipses[1]))
        (left, top, self.w, self.h) = cv2.boundingRect(np.array(points))
        left, top = left - EYE_BLUR, top - EYE_BLUR
        self.w, self.h = self.w + EYE_BLUR * 2, self.h + EYE_BLUR * 2
        self.img = self.img[top:top + self.h, left: left + self.w]
        self.mask = self.mask[top:top + self.h, left: left + self.w]

        # calculate the rest of params
        self.center, centers = centers2(eyes)

        self.length, self.angle = to_polar(centers[0], centers[1])
        self.center[0] -= left
        self.center[1] -= top

    def detect_eyes(self, img):
        """Detect eyes on image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        scale = w // self.scale
        gray = cv2.resize(gray, (self.scale, h // scale))

        rects = self.detector(gray, 1)
        # we assume only one face
        if len(rects) != 1:
            return None

        shape = self.predictor(gray, rects[0])
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(0, 68)]
        shape = np.array(shape) * scale

        return [
            shape[RIGHT_EYE_RANGE[0]:RIGHT_EYE_RANGE[1]],
            shape[LEFT_EYE_RANGE[0]:LEFT_EYE_RANGE[1]]
        ]

    def open_eyes(self, img):
        """Replace eyes with opened."""
        eyes = self.detect_eyes(img)
        if eyes is None:
            return img

        center, centers = centers2(eyes)
        length, angle = to_polar(centers[0], centers[1])

        w, h = self.w, self.h
        m = cv2.getRotationMatrix2D(
            tuple(self.center),
            self.angle - angle,
            length / self.length)
        patch = cv2.warpAffine(self.img, m, (w, h))
        patch_mask = cv2.warpAffine(self.mask, m, (w, h))
        patch_mask = cv2.merge(3 * [patch_mask])

        x = int(round(center[0] - self.center[0]))
        y = int(round(center[1] - self.center[1]))
        patched = patch.astype(float) * patch_mask / 255
        x1, y1 = x + w, y + h
        if x < 0 or y < 0 or y1 > img.shape[1] or x1 > img.shape[0]:
            return img

        patched += img[y:y1, x:x1].astype(float) * (255 - patch_mask) / 255
        opn = img.copy()
        opn[y:y1, x:x1] = patched.astype(np.uint8)

        return opn


class CapStream:
    """Launch the thread to read cam async."""

    def __init__(self, cap):
        """Read first frame."""
        self.cap = cap
        (self.ret, self.frame) = self.cap.read()
        self.stopped = False
        self.count = 1

    def start(self):
        """Start the thread."""
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        """Loop infinitely until the thread is stopped."""
        while True:
            if self.stopped:
                return
            (ret, frame) = self.cap.read()
            if ret is not True:
                continue
            self.frame = frame
            self.count += 1

    def read(self, count):
        """Return the frame if there are new."""
        if count == self.count:
            return count, None

        return self.count, self.frame

    def stop(self):
        """Stop."""
        self.stopped = True


def _init_pair(source, dest):
    # Grab the webcam feed and get the dimensions of a frame
    cap_source = cv2.VideoCapture(source)
    width = int(cap_source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    channels = 3
    print('Source size {}x{}'.format(width, height))

    # Set up the formatting of our loopback device - boilerplate
    format = v4l2.v4l2_format()
    format.type = v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT

    dest_cap = open(dest, 'wb')
    if (fcntl.ioctl(dest_cap, v4l2.VIDIOC_G_FMT, format) == -1):
        print("Unable to get video format data.")
        return -1, None, None

    format.fmt.pix.field = v4l2.V4L2_FIELD_NONE
    format.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_BGR24
    format.fmt.pix.width = width
    format.fmt.pix.height = height
    # format.fmt.pix.bytesperline = width * channels
    format.fmt.pix.sizeimage = width * height * channels

    result = fcntl.ioctl(dest_cap, v4l2.VIDIOC_S_FMT, format)

    return result, cap_source, dest_cap


def _main_loop(eye_contact, stream, dest):
    count = 0
    while True:
        count, frame = stream.read(count)
        if frame is None:
            continue
        s = perf_counter()
        frame = eye_contact.open_eyes(frame)
        e = perf_counter()
        fps = round(1. / (e - s), 1)
        if count % SHOW_FPS_EVEVRY == 0:
            print('Eye opener FPS {}'.format(fps), end='\r')  # noqa
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dest.write(frame)


if __name__ == '__main__':
    desc = """Eye contact

    Sample usage:
        python3 eye_contact.py open.jpg /dev/video0 /dev/video1
    """

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=desc)

    p.add_argument('open', type=str, help='open eye frame')
    p.add_argument('source', type=str, help='video device ex. /dev/vidoe0')
    p.add_argument('dest', type=str, help='video device destination')
    p.add_argument(
        '--scale', type=int, default=DEFAULT_SCALE,
        help='scale image on face detection')
    args = p.parse_args()

    r, src, dest = _init_pair(args.source, args.dest)
    print('Started ({} == 0). Press ctrl+c to exit.'.format(r))

    open_img = cv2.imread(args.open)
    eye_contact = EyeContact(open_img, args.scale)

    stream = CapStream(src).start()
    try:
        _main_loop(eye_contact, stream, dest)
    except KeyboardInterrupt:
        print('Exiting')

    stream.stop()
    src.release()
    dest.close()
