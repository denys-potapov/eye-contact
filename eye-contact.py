"""Fake eye contact."""
import cv2
import sys
import dlib
import numpy as np
import math


MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'

RIGHT_EYE_RANGE = (36, 42)
LEFT_EYE_RANGE = (42, 48)

EYE_SCALE = 2.0
EYE_BLUR = 21


def scale_box2d(box, scale):
    """Scale box."""
    h, w = box[1]
    return (box[0], (h * scale, w * scale), box[2])

# def to_polar(p1, p2):
#     """get a length and angel"""

#     dx = p2[0] p1[0]
class EyeContact:
    """EyeContact."""

    def __init__(self, img):
        """Init the eyes replacer."""
        self.img = img
        self.mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(MODEL_PATH)

        self.centers = []
        elipses = []
        for eye in self.detect_eyes(img):
            self.centers.append(eye.mean(axis=0))

            ellipse = cv2.fitEllipse(eye)
            mask_ellipse = scale_box2d(ellipse, EYE_SCALE)
            elipses.append(mask_ellipse)
            cv2.ellipse(self.mask, mask_ellipse, 255, -1)
        self.mask = cv2.GaussianBlur(self.mask, (EYE_BLUR, EYE_BLUR), 0)

        # crop everythong
        points = list(cv2.boxPoints(elipses[0])) + list(cv2.boxPoints(elipses[1]))
        (left, top, w, h) = cv2.boundingRect(np.array(points))
        left, top = left - EYE_BLUR, top - EYE_BLUR
        w, h = w + EYE_BLUR * 2, h + EYE_BLUR * 2
        self.img = self.img[top:top + h, left: left + w]
        self.mask = self.mask[top:top + h, left: left + w]

        self.centers = [(x - left, y - top) for (x, y) in self.centers]
        print(self.centers)
        # 
        # ]

    def detect_eyes(self, img):
        """Detect eyes on image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        # we assume only one face
        if len(rects) != 1:
            return []

        shape = self.predictor(gray, rects[0])
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(0, 68)]
        shape = np.array(shape)

        return [
            shape[RIGHT_EYE_RANGE[0]:RIGHT_EYE_RANGE[1]],
            shape[LEFT_EYE_RANGE[0]:LEFT_EYE_RANGE[1]]
        ]

    def open_eyes(self, img):
        """Replace eyes with opened."""
        pass

open_img = cv2.imread(sys.argv[1])

eye_contact = EyeContact(open_img)

cv2.imshow("Output", eye_contact.img)
cv2.imshow("mask", eye_contact.mask)
cv2.waitKey(10000)
