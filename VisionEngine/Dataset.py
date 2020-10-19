from Constants import CHARGED_WIDTH, FULL_PHOTOS_DATA_DIR, CROPPED_FACES_DATA_DIR
from cv2 import imwrite, imread, cvtColor, COLOR_RGB2BGR
from imutils import resize
from VisionEngine.Detector import Detector
from VisionEngine.OpenCVFaceDetector import OpenCVFaceDetector
import os

class Dataset:
    def __init__(self, searched_width=CHARGED_WIDTH, full_images_data_dir=FULL_PHOTOS_DATA_DIR,
                 cropped_faces_data_dir=CROPPED_FACES_DATA_DIR, faces_h=None, faces_w=None, verbose=True,
                 charge_full_dataset=False):
        self.data_dir = full_images_data_dir
        self.cropped_faces_data_dir = cropped_faces_data_dir
        self.verbose = verbose
        self.searched_width = searched_width
        if not os.path.exists(self.cropped_faces_data_dir) or \
                os.listdir(self.data_dir) != os.listdir(self.cropped_faces_data_dir):
            self.dataset = self.get_all_images(images_dir=self.data_dir, width=searched_width)
            self.generate_cropped_dataset(detector=Detector(OpenCVFaceDetector()))
        elif charge_full_dataset:
            self.dataset = self.get_all_images(images_dir=self.data_dir, width=searched_width)

        self.faces_dataset = self.get_all_images(images_dir=self.cropped_faces_data_dir, width=faces_h, height=faces_w)

    def get_all_images(self, images_dir, width=None, height=None):
        data = {}
        # loop over the image paths
        for person in os.listdir(images_dir):
            if self.verbose:
                print("Processing {name}".format(name=person))
            images = []
            sub_dirs = [os.path.join(images_dir, person)]
            while(len(sub_dirs) > 0):
                person_dir = sub_dirs.pop(0)
                for image_name in os.listdir(person_dir):
                    if os.path.isfile(os.path.join(person_dir, image_name)):
                        image = read_image(height=height, width=width, person_dir=person_dir, image_name=image_name)
                        images.append(image)
                    else:
                        sub_dirs.append(image_name)
            data[person] = images
        return data

    def generate_cropped_dataset(self, detector):
        for person, images in self.dataset.items():
            path = os.path.join(self.cropped_faces_data_dir, person)
            if not os.path.exists(path):
                os.makedirs(path)
            for image in images:
                image_path = os.path.join(path, str(len(os.listdir(path)))+'.jpg')
                faces = detector.crop_boxes_content(image=image)
                if len(faces)==1:
                    imwrite(image_path, faces[0])
                else:
                    print("Problem in image {path}. {faces} faces detected".format(path=image_path, faces=len(faces)))

def read_image(height, width, person_dir, image_name):
    # It gets the image with channels swapped to BGR
    image = imread(os.path.join(person_dir, image_name))
    if width is not None or height is not None:
        image = resize(image, width=width, height=height)
    return image