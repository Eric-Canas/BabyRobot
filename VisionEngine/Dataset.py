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
        """
        Reads the dataset, charges the images and allows to access to them with their respective labels.
        :param searched_width: Int. Width to which resize the complete images (without loosing aspect ratio)
        :param full_images_data_dir: Path. Dir with the complete images (from which crop the face).
        :param cropped_faces_data_dir: Path. Dir with the images of the faces.
        :param faces_h: Int. Height in pixels to which resize the faces of images (loosing aspect ration).
        :param faces_w: Int. Width in pixels to which resize the faces of images (loosing aspect ration).
        :param verbose: Boolean. If True, it verboses the process.
        :param charge_full_dataset: Boolean. If True charges the full dataset with the complete photos.
                                             If False only charges the dataset with the cropped faces.
        """
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
        """
        Return a dictionary with the list of all images for every identity in the dataset.
        :param images_dir: Path. Dir with the images.
        :param width: Int. Width to which resize the images.
        :param height: Int. Height to which resize the images.
        :return: Dictionary {Identity : List}. Dictionary with the list of all images for every identity in the dataset.
        """
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
                        try:
                            image = read_image(height=height, width=width, person_dir=person_dir, image_name=image_name)
                            images.append(image)
                        except:
                            continue
                    else:
                        sub_dirs.append(image_name)
            data[person] = images
        return data

    def generate_cropped_dataset(self, detector):
        """
        Uses a face detector for generating a face dataset from an original dataset of pictures containing people.
        :param detector: Detector. Face detector to use for detecting faces in the pictures of people.
        """
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
    """
    Function for reading an image and resizing it
    :param height: Int. Height to which resize the images.
    :param width: Int. Width to which resize the images.
    :param person_dir: String. Name of the person (folder) to which charge the image.
    :param image_name: String. Filename of the image to charge.
    :return: Numpy. The image charged in BGR format.
    """
    # It gets the image with channels swapped to BGR
    image = imread(os.path.join(person_dir, image_name))
    if width is not None or height is not None:
        image = resize(image, width=width, height=height)
    return image