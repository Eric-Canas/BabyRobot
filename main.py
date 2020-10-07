from RecognitionPipeline import RecognitionPipeline
from VisionEngine.Recognizer import train_recognizer
from PiCamera.PiCamera import PiCamera, capture_dataset
from VisionEngine.Dataset import Dataset

MODES = ("CAPTURE_NEW_DATASET", "TRAIN_RECOGNIZER", "SHOW_DETECTIONS_DEMO")

mode = "CAPTURE_NEW_DATASET"

if mode.upper() == "CAPTURE_NEW_DATASET":
    # Start to capture images until "q" is clicked
    capture_dataset(pipeline=RecognitionPipeline())

elif mode.upper() == "TRAIN_RECOGNIZER":
    # Train the final face recognizer
    train_recognizer(dataset=Dataset().faces_dataset)
elif mode.upper() == "SHOW_DETECTIONS_DEMO":
    # Start to capture images and to show detections in them
    pipeline = RecognitionPipeline()
    with PiCamera() as camera:
        for frame in camera.capture_continuous():
            pipeline.show_detections(image=frame)


