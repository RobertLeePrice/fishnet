import os
import glob
import cv2
import nd2
from multiprocessing import Pool
from pipeline import Pipeline
from nodes.common import * 
from nodes.io import ImageWrapper 
from nodes.preprocessing import *
import bigfish.stack as stack



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


def load_lut(filename):
    """Loads a LUT color map in RGB format."""
    lut = np.loadtxt(filename, dtype=np.uint8, delimiter=',', skiprows=1)
    return np.reshape(lut[:, 1:], (1, 256, 3))


def to_multichannel(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_bgr.astype(np.uint8)


def get_size(img, units='MB') -> float:
    if units == 'MB':
        return (img.size * img.itemsize) / 1000000
    else:
        print('Unit conversion not supported.')



def get_min_and_max(img, saturated=0.35):
    print('===== FUNCTION get_min_and_max =====')


    histogram, bins = np.histogram(img.flatten(), bins=256, range=(img.min(), img.max()))

    hmin, hmax = 0, 255
    threshold = int(img.size * saturated / 200.0)
    print('threshold: ', threshold)
    count = 0
    for i in range(len(histogram)):
        count += histogram[i]
        if count > threshold:
            hmin = i
            break
    count = 0
    for i in range(len(histogram)-1, -1, -1):
        count += histogram[i]
        if count > threshold:
            hmax = i
            break

    bin_size = bins[1] - bins[0]

    low_value = bins[hmin + 1] + hmin * bin_size
    high_value = bins[hmax + 1] + hmax * bin_size

    return low_value, high_value


def auto_contrast(img):
    """
    Apply auto-contrast adjustment to an image.
    """

    # low_value = 1278.0
    # high_value = 5931.0

    # low_value = 1254.84375
    # high_value = 5970.1875

    low_value, high_value = get_min_and_max(img)

    print('low_value: ', low_value)
    print('high_value: ', high_value)

    # low_value = 1200
    # high_value = 9600

    # Scale the image to 0-255 range
    scaled_image = (img - low_value) * (255.0 / (high_value - low_value))
    scaled_image[scaled_image < 0] = 0
    scaled_image[scaled_image > 255] = 255
    scaled_image = scaled_image.astype(np.uint8)

    return scaled_image

  
if __name__ == '__main__':
    folder = 'input'
    filenames = glob.glob(os.path.join(folder, '*.nd2'))

    blue = load_lut('blue.lut')
    yellow = load_lut('yellow.lut')

    img = nd2.imread(filenames[0])

    channel = 2
    c1 = img[0, channel, :, :]


    c1_adjusted = auto_contrast(c1)

    # c1_norm = cv2.normalize(c1, None, 0, 255, cv2.NORM_MINMAX)
    

    multi = to_multichannel(c1_adjusted)

    if channel == 0:
        lut = blue 
    else:
        lut = yellow

    dstImage = cv2.LUT(multi, cv2.cvtColor(lut, cv2.COLOR_BGR2RGB))



    # rna_2d_stretched = contrast_enhancer(c1)
    # # c1_norm2 = cv2.normalize(rna_2d_stretched, None, 0, 255, cv2.NORM_MINMAX)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # clahe = cv2.createCLAHE(clipLimit=5)
    # c2 = clahe.apply(c1)
    # c1_norm2 = cv2.normalize(c2, None, 0, 255, cv2.NORM_MINMAX)
    # multi2 = to_multichannel(c1_norm2)
    # dstImage2 = cv2.LUT(multi2, cv2.cvtColor(yellow, cv2.COLOR_BGR2RGB))
    # imS1 = cv2.resize(dstImage2, (960, 960)) 
    # cv2.imshow('Clahe', imS1)
    # cv2.waitKey(0)




    # cv2.imshow('Original', c1)
    # cv2.waitKey(0)

    # cv2.imshow('Normalized', c1_norm)
    # cv2.waitKey(0)

    # cv2.imshow('Multichannel', multi)
    # cv2.waitKey(0)
    

    imS = cv2.resize(dstImage, (960, 960)) 
    cv2.imshow('LUT2', imS)
    cv2.waitKey(0)





    #### STACKED

    channel_2 = 0
    z_stack = img[:, channel_2, :, :]

    # Create the maximum intensity projection image
    mip = np.min(z_stack, axis=0)
    # mip = z_stack

    # Convert the image to uint8 data type and scale to the 0-255 range
    # multi2 = to_multichannel(mip)

    # if channel_2 == 0:
    #     lut2 = blue 
    # else:
    #     lut2 = yellow

    # dstImage2 = cv2.LUT(multi2, cv2.cvtColor(lut2, cv2.COLOR_BGR2RGB))

    mip = cv2.normalize(mip, None, 0, 255, cv2.NORM_MINMAX)
    mip = mip.astype(np.uint8)

    imS2 = cv2.resize(mip, (960, 960))                    # Resize image

    cv2.imshow('Channel Stack', imS2)
    cv2.waitKey(0)

    cv2.destroyAllWindows()





    # print(filenames)

    # nodes = [
    #     Preprocessing([
    #         # ImageNormalization()
    #         CLAHEContrast()
    #         # LinearContrast(),
    #     ]),
    #     # MachineLearning('model'),
    #     # PostProcessing([])
    # ]
    # pipeline = Pipeline(nodes)
    
    # with Pool() as p:
    #     p.starmap(process_image, [(filename, pipeline) for filename in filenames])






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
