import os
import glob
import cv2
from multiprocessing import Pool
from pipeline import Pipeline
from nodes.common import * 
from nodes.io import ImageWrapper 
from nodes.preprocessing import *


def process_image(filename, pipeline):

    print('===== FUNCTION process_image =====')

    # load image from file
    img = ImageWrapper(filename)
    
    print('After Image Loaded')

    print('\nImage: ', img.image)

    print(type(img.image))
    cv2.imshow(img.filename, img.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
    # process image through pipeline
    processed_image = pipeline.process(img)

    print('After Processing')
    cv2.imshow('test', processed_image.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # save processed image to file
    pass
    
if __name__ == '__main__':
    folder = 'input'
    filenames = glob.glob(os.path.join(folder, '*.tif'))

    print(filenames)

    nodes = [
        Preprocessing([
            CLAHEContrast()
            # LinearContrast(),
        ]),
        # MachineLearning('model'),
        # PostProcessing([])
    ]
    pipeline = Pipeline(nodes)
    
    with Pool() as p:
        p.starmap(process_image, [(filename, pipeline) for filename in filenames])






# TODO: Build Pipeline from Config

# def load_config(config_file):

#     # print('===== FUNCTION load_config =====')

#     with open(config_file) as f:
#         config = json.load(f)
    
#     print(config)

#     steps = []
#     for step_config in config["steps"]:
#         # print(step_config)

#         module_name = importlib.import_module(step_config["module"])
#         # print(module_name)

#         class_name = getattr(module_name, step_config["class"])
#         # print(class_name)

#         parameters = step_config.get("parameters")
#         # print(parameters)

#         instance = class_name(**parameters)
#         # print(instance)

#         steps.append({"name": step_config["name"], "instance": instance})
    
#     config['steps'] = steps

#     print(config)

#     return config

# # Parse command-line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--config", "-c", 
#     default="config.json", 
#     help="configuration file"
# )
# args = parser.parse_args()

# # Load the steps from the configuration file
# steps = load_config(args.config)

# # Build the pipeline
# pipeline = Pipeline()
# for step in steps:
#     print(step)
#     print(step.get('name'))
#     # print(f'Building Step {step.get("name")}')
#     pipeline.add_step(step)
