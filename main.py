from RecognitionPipeline import RecognitionPipeline
from VisionEngine.Recognizer import train_recognizer
from RobotController.PiCamera.PiCamera import PiCamera, capture_dataset
from VisionEngine.Dataset import Dataset
from RobotController.ReinforcementLearningController.Trainer import Trainer
from RobotController.ReinforcementLearningController.World import World
from RobotController.AddOnControllers.Controller import Controller
from VisionEngine.Embedding import Embedding
from VisionEngine.MobileFaceNet import MobileFaceNet
from RobotController.ReinforcementLearningController.Validator import Validator
from Constants import DEFAULT_PERSON_TO_FOLLOW

MODES = ("CAPTURE_NEW_DATASET", "TRAIN_RECOGNIZER", "SHOW_DETECTIONS_DEMO", "TRAIN_MOVEMENT", "PLAY")

mode = "SHOW_DETECTIONS_DEMO"

if mode.upper() == "CAPTURE_NEW_DATASET":
    # Start to capture images until "q" is clicked
    capture_dataset(pipeline=RecognitionPipeline())

elif mode.upper() == "TRAIN_RECOGNIZER":
    embedder = Embedding(MobileFaceNet())
    # Train the final face recognizer
    train_recognizer(dataset=Dataset().faces_dataset, embedder=embedder)

elif mode.upper() == "SHOW_DETECTIONS_DEMO":
    # Start to capture images and to show detections in them
    pipeline = RecognitionPipeline()
    with PiCamera() as camera:
        for frame in camera.capture_continuous():
            pipeline.show_detections(image=frame)

elif mode.upper() == "TRAIN_MOVEMENT":
    person_to_follow = DEFAULT_PERSON_TO_FOLLOW
    showing = True
    pipeline = RecognitionPipeline()
    controller = Controller()
    env = World(objective_person=person_to_follow, controller=controller, recognition_pipeline=pipeline,
                average_info_from_n_images=1)
    trainer = Trainer(env=env)
    trainer.train(show=True)
elif mode.upper() == "PLAY":
    person_to_follow = DEFAULT_PERSON_TO_FOLLOW
    showing = True
    pipeline = RecognitionPipeline()
    controller = Controller()
    env = World(objective_person=person_to_follow, controller=controller, recognition_pipeline=pipeline,
                average_info_from_n_images=1)
    validator = Validator(person_to_follow=person_to_follow)
    validator.validate(show=True)
else:
    raise NotImplementedError()


