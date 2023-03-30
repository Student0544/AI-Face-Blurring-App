import cv2
import os
import numpy as np
import csv


def vector(csv_file):
    """ Turns the CSV file into another CSV file composed of multiple vectors"""
    with open(csv_file, "r", newline="") as f:
        boxes = []

        file = csv.reader(f)
        # next(file)

        image = []
        for line in file:
            if line:
                image += line

            elif not line:
                boxes.append(image)
                image = []

        with open(csv_file, "w", newline="") as f:
            file = csv.writer(f)
            for line in boxes:
                file.writerow(line)


def find_coordinates(folder_path: str, margin: int, minimum: int, *target_colors: list):

    """Returns a CSV file containing the coordinates corresponding to the colors' position. Each row corresponds to
    the position of one individual object, and a blank row means that it's the coordinates of the next image.
    If multiple target colors are entered, then the contours will treat those overlapping colors as one
    object. Please enter BGR values for the colors. Margin is the variation in BGR value allowed. Minimum is the
    smallest allowed size of a face in pixels"""

    errors = 0

    # Directory
    input_folder = rf"{folder_path}"
    directory = os.path.basename(input_folder)
    new_path = rf"{input_folder} head coordinates\{directory} coordinates.csv"
    os.makedirs(rf"{input_folder} head coordinates", exist_ok=True)
    #

    # Colors
    colors = list(map(np.array, target_colors))
    #

    # Finding Contours of all Images
    for img_name in os.listdir(input_folder):

        try:
            filepath = os.path.join(input_folder, img_name)
            image = cv2.imread(filepath)

            # Turn the Image into a Binary Image Based on the Target Colors
            masks = [cv2.inRange(image, array-margin, array+margin) for array in colors]

            if len(masks) > 1:
                for index in range(len(masks)):
                    if index == 0:
                        mask = cv2.bitwise_or(masks[0], masks[1])

                    elif index < len(masks)-1:
                        mask = cv2.bitwise_or(mask, masks[index+1])

            else:
                mask = masks[0]

            # Turn the Image into a Binary Image. Also smooths out noises and small spots.
            binary_image = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)[1]

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

            contours = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contours = [np.array(contour) for contour in contours]

            coordinates = []  # The format of the coordinates will be (xi, yi, xf, yf)
            for contour in contours:
                cx = [row[0][0] for row in contour]
                cy = [row[0][1] for row in contour]
                coordinate = (min(cx), min(cy), max(cx), max(cy))
                area = (coordinate[2] - coordinate[0])*(coordinate[3]-coordinate[1])

                if area >= minimum:
                    coordinates.append(coordinate)
                else:
                    print(f"Ignored a face with size {area} in image {img_name}.")

            with open(new_path, "a", newline="") as file:
                obj = csv.writer(file)

                # if img_name == os.listdir(input_folder)[0]:
                #     obj.writerow(("xi", "yi", "xf", "yf"))

                if len(coordinates) > 0:
                    for row in coordinates:
                        if row is not None:
                            obj.writerow(row)
                else:
                    print(f"No faces were found in image {img_name}")
                    obj.writerow("-")

                obj.writerow([])

            if img_name in os.listdir(input_folder)[50:60]:

                copy1 = image.copy()
                copy2 = image.copy()
                copy3 = image.copy()
                copy3[:, :] = [0, 0, 0]

                for coordinate in coordinates:

                    # cv2 takes ranges of pixels in the format of [yi:yf, xi:xf]
                    region = copy1[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]]

                    blur = cv2.GaussianBlur(region, (31, 31), 0, 0)
                    copy1[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]] = blur
                    copy2[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]] = [255, 255, 255]
                    copy3[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]] = [255, 255, 255]

                edit = cv2.hconcat([image, copy1, copy2, copy3])

                if edit.shape[1] > 2000:
                    new_height = int(edit.shape[0]/edit.shape[1] * 2000)
                    edit = cv2.resize(edit, (new_height, 2000))

                cv2.imshow("First Edits", edit)
                cv2.waitKey(0)
                cv2.destroyAllWindows()



        except ValueError:
            errors += 1

    # vector(new_path)

    plural_noun = {True: "", False: "s"}[errors < 1]
    plural_verb = {True: "was", False: "were"}[errors < 1]
    print(f"{len(os.listdir(input_folder)) - errors} image{plural_noun} processed. "
          f"{errors} image{plural_noun} {plural_verb} unsuccessfully processed.")


face_color = [196, 196, 196]
glasses_color = [61, 61, 61]
margin_of_error = 0
minimum_head_size = 40

find_coordinates(r"path_to_segmented_image_folder_here", margin_of_error, minimum_head_size,
                  face_color, glasses_color)
vector(r"path_to_new_csv_created_from_the_above_here")
