import cv2
import os
from find_color_coordinates import find_coordinates, vector


def process_image(image_path: str, new_height: int, new_width: int, brighten: float = 1, contrast: float = 1,
                 flip: int = None, rotate: int = None, rename: str = None):
    """Outputs another folder where all images are resized to the set dimensions.
    Make sure the path you input is the root path reaching the input image. Height and Width in pixels. Possible values
    for flip are 0, 1, and -1. 0 will flip the image vertically, 1 will flip the image horizontally, and -1 will flip
    the image vertically and horizontally. Rotate will rotate the image in multiples of 90 degrees counterclockwise.
    Please enter an integer between 1 and 3 inclusive."""

    errors = 0

    # Image Dimensions
    h = new_height
    w = new_width
    #

    # Directory
    input_folder = rf"{image_path}"
    os.makedirs(rf"{input_folder} processed", exist_ok=True)
    #

    space = {True: " ", False: ""}[rename is not None]

    # Reading and Processing all Images
    for (index, img_name) in enumerate(os.listdir(input_folder)):
        try:
            filepath = os.path.join(input_folder, img_name)
            image = cv2.imread(filepath)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

            if (contrast, brighten) != (1, 1):
                image = cv2.convertScaleAbs(image, alpha=brighten, beta=contrast)

            if flip in (0, 1, -1):
                image = cv2.flip(image, flip)

            if rotate in (1, 2, 3):
                if rotate == 1:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotate == 2:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                else:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            if img_name == os.listdir(input_folder)[0]:
                cv2.imshow("First Edit", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if rename:
                cv2.imwrite(rf"{input_folder} processed/{rename}{space}{index}.png", image)

            else:
                cv2.imwrite(rf"{input_folder} processed/{img_name}", image)

        except TypeError:
            errors += 1

    plural_noun = {True: "", False: "s"}[errors < 1]
    plural_verb = {True: "was", False: "were"}[errors < 1]
    print(f"{len(os.listdir(input_folder)) - errors} image{plural_noun} processed. "
          f"{errors} image{plural_noun} {plural_verb} unsuccessfully processed.")


input_folder1 = r"path_to_normal_images_here"
input_folder2 = r"path_to_segmented_images_corresponding_to_the_above_here"
new_height = 300
new_width = 300
brighten_factor = 15
contrast_factor = 1

# To process the images normally, run the first line below for the normal camera images, and run the second line for
# the segmented images
process_image(input_folder1, new_height, new_width)
process_image(input_folder2, new_height, new_width, brighten=brighten_factor)
