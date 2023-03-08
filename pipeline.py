class Pipeline:
    def __init__(self, nodes):
        self.nodes = nodes

    def add_node(self, node):
        self.nodes.append(node)
        
    def process(self, image):
        for node in self.nodes:
            image = node.process(image)
        return image
