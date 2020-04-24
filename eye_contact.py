"""Fake eye contact."""
import cv2
import sys
import dlib
import numpy as np
import math


MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'

RIGHT_EYE_RANGE = (36, 42)
LEFT_EYE_RANGE = (42, 48)

EYE_SCALE = 1.5
EYE_BLUR = 21

DEFAULT_SCALE = 320


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
            return []

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
        center, centers = centers2(eyes)
        length, angle = to_polar(centers[0], centers[1])

        w, h = self.w, self.h
        m = cv2.getRotationMatrix2D(
            tuple(self.center),
            self.angle - angle,
            self.length / length)
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

if __name__ == '__main__':
    """Not sure it's needed here."""
    open_img = cv2.imread(sys.argv[1])
    test_img = cv2.imread(sys.argv[2])

    eye_contact = EyeContact(open_img)

    cv2.imshow("Output", eye_contact.img)
    result = eye_contact.open_eyes(test_img)
    cv2.imwrite(sys.argv[2] + '.result.jpg', result)
    cv2.imshow("Result", result)
    cv2.waitKey(10000)
