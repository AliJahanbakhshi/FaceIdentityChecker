# Face Identity Checker

A real-time face recognition and identity verification system built with **Python and OpenCV**.

## Features

* Real-time face detection
* Face recognition using **SFace**
* Face detection using **YuNet**
* Webcam support
* Adjustable recognition sensitivity
* Real-time match score

## Requirements

```bash
pip install opencv-python
```

## Usage

Place these files in the project folder:

```text
main.py
name.png
face_detection_yunet_2023mar.onnx
face_recognition_sface_2021dec.onnx
```

Run the program:

```bash
python main.py
```

Enter a sensitivity value between `0.0` and `1.0`.

Press **Q** to exit.

## Technologies

* Python
* OpenCV
* YuNet
* SFace
* ONNX

## Note

This project is created for **educational and experimental purposes**.
