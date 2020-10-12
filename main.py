from RecognitionPipeline import RecognitionPipeline
from VisionEngine.Recognizer import train_recognizer
from RobotController.PiCamera.PiCamera import PiCamera, capture_dataset
from VisionEngine.Dataset import Dataset
from RobotController.ReinforcementLearningController.Trainer import Trainer
from RobotController.ReinforcementLearningController.World import World
from RobotController.AddOnControllers.Controller import Controller

MODES = ("CAPTURE_NEW_DATASET", "TRAIN_RECOGNIZER", "SHOW_DETECTIONS_DEMO", "TRAIN_MOVEMENT")

mode = "TRAIN_MOVEMENT"

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

elif mode.upper() == "TRAIN_MOVEMENT":
    person_to_follow = 'Eric'
    showing = True
    pipeline = RecognitionPipeline()
    controller = Controller()
    env = World(objective_person=person_to_follow, controller=controller, recognition_pipeline=pipeline,
                average_info_from_n_images=5)
    trainer = Trainer(env=env)
    trainer.train()

else:
    raise NotImplementedError()


