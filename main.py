import cv2
import os

# Files
REF = "name.png"
YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"

# Ask for sensitivity
try:
    THRESHOLD = float(input("Enter sensitivity (0.0 - 1.0): "))
    THRESHOLD = max(0.0, min(1.0, THRESHOLD))
except ValueError:
    THRESHOLD = 0.40

# Check files
for f in [REF, YUNET, SFACE]:
    if not os.path.exists(f):
        print(f"File not found: {f}")
        exit()

# Load models
detector = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.3, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(SFACE, "")

# Load reference face
reference = cv2.imread(REF)
if reference is None:
    print("Could not load PNG file")
    exit()

rh, rw = reference.shape[:2]
detector.setInputSize((rw, rh))
_, faces = detector.detect(reference)

if faces is None or len(faces) == 0:
    print("No face found in name.png")
    exit()

reference_face = faces[0]
aligned = recognizer.alignCrop(reference, reference_face)
reference_feature = recognizer.feature(aligned)

print(f"Ready! Sensitivity: {THRESHOLD:.2f}")

# Open camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera")
    exit()

while True:
    ret, frame = camera.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is None:
        cv2.putText(frame, "NO FACE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        for face in faces:
            x, y, fw, fh = map(int, face[:4])

            aligned = recognizer.alignCrop(frame, face)
            feature = recognizer.feature(aligned)

            score = recognizer.match(
                reference_feature,
                feature,
                cv2.FaceRecognizerSF_FR_COSINE
            )

            if score >= THRESHOLD:
                text, color = f"MATCH {score:.2f}", (0, 255, 0)
            else:
                text, color = f"NOT MATCH {score:.2f}", (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 3)
            cv2.putText(frame, text, (x, y + fh + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    cv2.imshow("Identity Check", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
