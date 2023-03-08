import os
import cv2
from .common import Node

VALID_EXTENSIONS = [".tif", ]


class ImageWrapper(Node):
    def __init__(self, filename):
        self.filename = filename
        self.image = self.load()
        
    def load(self):
        _, extension = os.path.splitext(self.filename)
        if extension in VALID_EXTENSIONS:
            return cv2.imread(self.filename, 0)
        else:
            raise ValueError("Unsupported file format")