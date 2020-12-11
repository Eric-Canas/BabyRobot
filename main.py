from RecognitionPipeline import RecognitionPipeline
from VisionEngine.Recognizer import train_recognizer
from RobotController.PiCamera.PiCamera import PiCamera, capture_dataset
from VisionEngine.Dataset import Dataset
from RobotController.MovementControl.ReinforcementLearningController.Trainer import Trainer
from RobotController.MovementControl.DQNWorld import World
from RobotController.AddOnControllers.Controller import Controller
from VisionEngine.Embedding import Embedding
from VisionEngine.MobileFaceNet import MobileFaceNet
from RobotController.MovementControl.ReinforcementLearningController.Validator import Validator
from Constants import DEFAULT_PERSON_TO_FOLLOW
from RobotController.AddOnControllers.MotorController import MotorController
from RobotController.RLConstants import MOVEMENT_TIME, SYNC_MODE
from RobotController.ClientServer.Socket import Socket
from RobotController.ClientServer.ServerPipeline import ServerPipeline, SAVE_FILE_CODE
from RobotController.ClientServer.ClientPipeline import ClientPipeline
from RobotController.MovementControl.FiniteStateMachine import FiniteStateMachine
from RobotController.MovementControl.FiniteStateMachineWorld import FiniteStateMachineWorld
import sys

MODES = ("CAPTURE_NEW_DATASET", "TRAIN_RECOGNIZER", "SHOW_DETECTIONS_DEMO", "TRAIN_MOVEMENT", "PLAY", "SERVER")
CONTROLLERS = ('DQN', 'FINITE_STATE_MACHINE')
controller = 'FINITE_STATE_MACHINE'
execution_mode = "PLAY"
movement_mode = SYNC_MODE
execute_on_server = True
teleoperated_exploration = True


if execution_mode.upper() == "CAPTURE_NEW_DATASET":
    # Start to capture images until "q" is clicked
    capture_dataset(pipeline=RecognitionPipeline())

elif execution_mode.upper() == "TRAIN_RECOGNIZER":
    embedder = Embedding(MobileFaceNet())
    # Train the final face recognizer
    train_recognizer(dataset=Dataset().faces_dataset, embedder=embedder)

elif execution_mode.upper() == "SHOW_DETECTIONS_DEMO":
    # Start to capture images and to show detections in them
    pipeline = RecognitionPipeline()
    with PiCamera() as camera:
        for frame in camera.capture_continuous():
            pipeline.show_recognitions(image=frame)

elif execution_mode.upper() == "TRAIN_MOVEMENT":

    ip = sys.argv[1] if len(sys.argv) > 1 and execute_on_server else None

    # Train following only one person (without recognition) for improving the training velocity in a 70%
    person_to_follow = None#DEFAULT_PERSON_TO_FOLLOW
    showing = False
    pipeline = RecognitionPipeline() if not execute_on_server else ClientPipeline(socket=Socket(client=True, ip=ip))
    controller = Controller(MotorController(default_movement_time=MOVEMENT_TIME, movement_mode=movement_mode))
    env = World(objective_person=person_to_follow, controller=controller, recognition_pipeline=pipeline,
                average_info_from_n_images=1, movement_mode=movement_mode)
    trainer = Trainer(env=env, tele_operate_exploration=teleoperated_exploration)
    trainer.train(show=showing)

elif execution_mode.upper() == "PLAY":
    person_to_follow = None#DEFAULT_PERSON_TO_FOLLOW
    showing = True
    pipeline = RecognitionPipeline()
    controller = Controller(MotorController(default_movement_time=MOVEMENT_TIME, movement_mode=movement_mode))
    if controller == 'DQN':
        env = World(objective_person=person_to_follow, controller=controller, recognition_pipeline=pipeline,
                    average_info_from_n_images=1, movement_mode=movement_mode)
        validator = Validator(person_to_follow=person_to_follow)
        validator.validate(show=showing)
    else:
        state_machine = FiniteStateMachine(controller=controller)
        env = FiniteStateMachineWorld(objective_person=person_to_follow, controller=controller, recognition_pipeline=pipeline,
                    average_info_from_n_images=1, movement_mode=movement_mode, finite_state_machine=state_machine)
        env.play()


elif execution_mode.upper() == 'SERVER':
    socket = Socket(client=False)
    server_pipeline = ServerPipeline(self_socket=socket)
    while True:
        function_to_execute = socket.receive_int_code()
        if function_to_execute == SAVE_FILE_CODE:
            socket.receive_file(save_it=True)
        else:
            server_pipeline.execute_and_send_result(code=function_to_execute, show=True)

else:
    raise NotImplementedError()


