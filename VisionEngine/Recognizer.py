from pickle import load, dump
import numpy as np
import os
from Constants import FACE_RECOGNIZER_FILE, FACE_RECOGNIZER_DIR,\
                        DECIMALS, KNOWN_NAMES

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from VisionEngine.Dataset import Dataset
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

from random import uniform
from skimage.util import random_noise
from skimage.transform import rotate

ROTATION_PROBABILITY = 0.33
NOISE_PROBABILITY = 0.8
HORIZONTAL_FLIP_PROBABILITY = 0.2

class Recognizer:
    def __init__(self, embedding, recognizer_path=os.path.join(FACE_RECOGNIZER_DIR,FACE_RECOGNIZER_FILE),
                 binary_recognition=None):
        self.embedding = embedding
        self.binary_recognition = binary_recognition
        if os.path.exists(recognizer_path):
            self.recognizer = load(open(recognizer_path, 'rb'))
        else:
            print("Training SVC...")
            self.recognizer = train_recognizer(dataset=Dataset().faces_dataset, embedder=self.embedding,
                                               binary_recognition=self.binary_recognition)

    def predict(self, x, as_name = True):
        y = self.recognizer.predict(x)
        if as_name:
            y = [KNOWN_NAMES[face_id] if face_id >=0 else 'Other' for face_id in y]
        return y

def train_recognizer(dataset, embedder, components_reduction=128, save_at=FACE_RECOGNIZER_DIR,
                     file_name=FACE_RECOGNIZER_FILE, binary_recognition=None, verbose=True):
    x_train, x_val, y_train, y_val = get_train_val_splits(dataset=dataset, embedder=embedder,
                                                          binary_recognition=binary_recognition)


    pipe = Pipeline(steps=[('PCA', PCA(n_components=components_reduction)),
                           ('MLP', MLPClassifier(hidden_layer_sizes=(256,128), max_iter=10000, validation_fraction=0.01))])
                           #('BinaryOutput', FunctionTransformer(np.sign))], verbose=True)
                           #('SVC', OneClassSVM(max_iter=50000))], verbose=True)
    """
    #pipe = LinearDiscriminantAnalysis()
    """
    """
    if binary_recognition is None:
        pipe = MLPClassifier(max_iter=10000)
    else:
        pipe = MLPRegressor(max_iter=10000)
    """
    pipe = pipe.fit(X=x_train, y=y_train)

    if verbose:
        print('Recognizer Pipeline Trained:\n'
              '\tTrain Acc: {TrainAcc}%\n'
              '\tVal Acc: {ValAcc}%'.format(TrainAcc=round(pipe.score(X=x_train, y=y_train)*100, ndigits=DECIMALS),
                                           ValAcc=round(pipe.score(X=x_val, y=y_val)*100, ndigits=DECIMALS)))
    if not os.path.exists(save_at):
        os.makedirs(save_at)
    else:
        [os.remove(os.path.join(save_at,file)) for file in os.listdir(save_at)]
    dump(pipe, open(os.path.join(save_at, file_name), 'wb'))
    save_confusion_matrix(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                          pipeline=pipe, save_at=save_at, binary_recognition=binary_recognition)
    return pipe

def save_confusion_matrix(x_train, y_train, x_val, y_val, pipeline,
                          save_at=FACE_RECOGNIZER_DIR, labels=KNOWN_NAMES,
                          binary_recognition=None):
    if binary_recognition is not None:
        labels = ['Other', binary_recognition]
    for set, (x, y) in [('Train', (x_train, y_train)), ('Val', (x_val, y_val))]:
        y_pred = pipeline.predict(x)
        if binary_recognition:
            y_pred = np.sign(y_pred)
            acc = np.mean(y == y_pred)
        acc = pipeline.score(X=x, y=y)
        y[y == -1], y_pred[y_pred==-1] = 0, 0
        ids = np.sort(np.unique(y))
        matrix = confusion_matrix(y_true=y, y_pred=y_pred, labels=ids, normalize='pred')
        plt.matshow(matrix, cmap=plt.get_cmap('YlOrRd'))
        plt.gca().xaxis.tick_bottom()
        plt.title('{set} Confusion Matrix: {acc}% Acc'.format(set=set, acc=round(acc*100,ndigits=DECIMALS)))
        plt.colorbar()
        plt.xticks(range(len(ids)), labels, rotation=60)
        plt.yticks(range(len(ids)), labels, rotation=60)
        plt.savefig(os.path.join(save_at, set+' - Acc_{acc}.png'.format(acc=round(acc*100, ndigits=DECIMALS))))

def get_train_val_splits(dataset, embedder, val_split=0.1, binary_recognition = None,
                         data_augmentation_times=25):
    X, Y = [], []
    for _ in range(data_augmentation_times):
        for i, (person, images) in enumerate(dataset.items()):
            Y.extend([i for _ in range(len(images))])
            X.extend([embedder.predict(data_augmentate(image)) for image in images])
    X, Y = np.array(X), np.array(Y)
    if binary_recognition is not None:
        binary_recognition = KNOWN_NAMES.index(binary_recognition)
        Y = (np.array(Y == binary_recognition, dtype=np.int) * 2) - 1
    return train_test_split(X, Y, test_size=val_split, random_state=612, stratify=Y)

def data_augmentate(image, rotation_probability=ROTATION_PROBABILITY, noise_probability=NOISE_PROBABILITY,
                    horizontal_flip_probability = HORIZONTAL_FLIP_PROBABILITY):
    if uniform(0,1) <= rotation_probability:
        image = random_rotation(image)
    if uniform(0,1) <= noise_probability:
        image = random_noise(image)
    if uniform(0,1) <= horizontal_flip_probability:
        image = image[:, ::-1]
    return image.astype(np.uint8)

def random_rotation(image):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = uniform(-60, 60)
    return rotate(image, random_degree)
