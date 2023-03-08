## For now these are just wrappers and don't do much. Might delete in the future.


class Node:
    def __init__(self):
        pass
        
    def process(self, image):
        pass


class Preprocessing(Node):
    def __init__(self, nodes):
        self.nodes = nodes
        
    def process(self, image):
        for node in self.nodes:
            image = node.process(image)
        return image


class MachineLearning(Node):
    def __init__(self, model):
        self.model = model
        
    def process(self, image):
        # apply machine learning model to image
        pass


class PostProcessing(Node):
    def __init__(self, nodes):
        self.nodes = nodes
        
    def process(self, image):
        for node in self.nodes:
            image = node.process(image)
        return image