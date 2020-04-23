"""Fake eye contact."""
import cv2
import sys
import dlib


MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'

RIGHT_EYE_RANGE = (36, 42)
LEFT_EYE_RANGE = (42, 48)


class EyeContact:
    """EyeContact."""

    def __init__(self, img):
        """Init the eyes replacer."""
        # dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(MODEL_PATH)

        eyes = self.detect_eyes(img)
        for eye in eyes:
            for (x, y) in eye:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    def detect_eyes(self, img):
        """Detect eyes on image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        # we assume only one face
        if len(rects) != 1:
            return

        shape = self.predictor(gray, rects[0])
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(0, 68)]

        return [
            shape[RIGHT_EYE_RANGE[0]:RIGHT_EYE_RANGE[1]],
            shape[LEFT_EYE_RANGE[0]:LEFT_EYE_RANGE[1]]
        ]

    def open_eyes(self, img):
        """Replace eyes with opened."""
        pass

open_img = cv2.imread(sys.argv[1])

eye_contact = EyeContact(open_img)

cv2.imshow("Output", open_img)
cv2.waitKey(10000)
