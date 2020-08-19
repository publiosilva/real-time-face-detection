import cv2


class FaceDetector:
    def __init__(self):
        self.trained_face_data = \
            cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

    def get_face_coordinates(self, frame):
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = \
            self.trained_face_data.detectMultiScale(grayscaled_img)

        return face_coordinates

    def detect(self):
        webcam = cv2.VideoCapture(0)

        while True:
            _, frame = webcam.read()
            face_coordinates = self.get_face_coordinates(frame)

            print("Detected Faces:", len(face_coordinates))

            for (x, y, w, h) in face_coordinates:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Face Detector (Press \"Q\" to Exit)", frame)
            key = cv2.waitKey(1)

            if key == 81 or key == 113:
                break

        webcam.release()


if __name__ == "__main__":
    fd = FaceDetector()
    fd.detect()
