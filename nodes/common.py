import time
## For now these are just wrappers and don't do much. Might delete in the future.



class Node:
    def __init__(self):
        pass
        
    # def process(self, image):
    #     pass

    # def process(self, image):
    #     start_time = time.time()
    #     # Call the implementation-specific processing code
    #     self._process(image)
    #     end_time = time.time()
    #     print(f"Processing took {end_time - start_time} seconds")


    def process(self, image):
        # This method should be implemented by the subclasses
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