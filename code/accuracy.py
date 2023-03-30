from torch import from_numpy, load, device, no_grad
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from cv2 import imread, cvtColor, COLOR_BGR2RGB
import csv
from os import listdir
from os.path import join
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

def csv_2_list(path):
    with open(path, "r", newline="") as f:
        file = csv.reader(f)
        targets = []

        g = lambda y: float(y)
        for line in file:

            if line == ["-"]:
                targets.append([])

            else:
                line = list(map(g, line))
                box = [line[4 * bracket:4 * bracket + 4] for bracket in range(len(line) // 4)]
                box = [tuple(b) for b in box]

                # arr.append((tensor(box), tensor(label)))
                targets.append(box)

    return targets

def censor_img_test(file, model, l_limit=0.1, u_limit=0.5, l_ratio=0.1, u_ratio=5, score=0.999):
    """l_limit and u_limit are in percentage. This will filter out blur boxes that cover an area that is
    outside the accepted percentage of area covered. l_ratio and u_ratio is the direct ratio between the blur box's
    width over height. This will filter out boxes that are not within the acceptable range of ratios. score is also in
    percentage, and this will filter out boxes where the model's confidence that a box covers a face is not higher than
    the set threshold."""

    model = model

    if True:
        image = imread(file)
        h, w = image.shape[0:2]
        width = w / 300
        height = h / 300

        # the following line of code converts the image from BGR to RGB channel format
        img = image.copy()
        # img = resize(img, (300, 300), interpolation=INTER_AREA)
        img = cvtColor(img, COLOR_BGR2RGB)
        img = from_numpy(img.T)
        ############################

        # Making the prediction
        with no_grad():
            output = model([img.float()])
        scores = output[0]['scores'].numpy()

        #############################

        # The following code removes blur boxes that are larger than a certain pixel amount and boxes where the model
        # is not above a certain confidence that it covers a face
        selected_scores = scores >= score
        l_limit = l_limit * (w*h)
        u_limit = u_limit * (w*h)
        l_ratio = l_ratio
        u_ratio = u_ratio
        ##################################################################################

        # Filters out the boxes based on the conditions set in the above
        selected_predictions = {key: value[selected_scores] for key, value in output[0].items()}
        boxes = selected_predictions['boxes'].tolist()
        b = []
        # v = []
        for LIST in boxes:
            box = list(map(int, LIST))
            if (box[0] != box[2]) and (box[1] != box[3]):
                b.append(list(map(int, LIST)))
                # v.append(True)
            else:
                # v.append(False)
                pass
        # v = array(v)
        # mask = array([((u_limit >= (box[2] - box[0]) * (box[3] - box[1]) * width * height >= l_limit)
        #                and (l_ratio <= ((box[2] - box[0]) * width) / ((box[3] - box[1]) * height) <= u_ratio)) for box in b])
        b = [(int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)) for box in b
             if (u_limit >= (box[2] - box[0]) * (box[3] - box[1]) * width * height >= l_limit)
             and (l_ratio <= ((box[2] - box[0]) * width) / ((box[3] - box[1]) * height) <= u_ratio)]
        ###########################################

        # Information about the boxes that were not filtered out
        # print(len(b), "boxes remain.")
        # print(b)
        # remaining_scores = scores[selected_scores][v][mask].tolist()
        # print("Confidence Score:", remaining_scores)
        # print("Average Score:", round(sum(remaining_scores)/len(remaining_scores), 8),
        #       f"(Lowest: {round(min(remaining_scores), 8)}, Highest: {round(max(remaining_scores), 8)})")
        # percentages = [round((box[2]-box[0])*(box[3]-box[1])/(w*h), 3) for box in b]
        # print("Percentage Occupied:", percentages)
        # print("Average Percentage:", round(sum(percentages)/len(percentages), 4),
        #       f"(Lowest: {round(min(percentages), 4)}, Highest: {round(max(percentages), 4)})")
        # ratios = [round((box[2]-box[0])/(box[3]-box[1]), 4) for box in b]
        # print("Ratio of Boxes (W/H):", ratios)
        # print("Average Ratio:", round(sum(ratios)/len(ratios), 4),
        #       f"(Lowest: {round(min(ratios), 4)}, Highest: {round(max(ratios), 4)})", "\n")
        ###################################################

        return b

    # except:
    #     print("error")
    #     return "An error occurred while processing your file."

def get_area(coordinates:list):
    polygons = [Polygon([(xi, yi), (xi, yf), (xf, yf), (xf, yi)]) for xi, yi, xf, yf in coordinates]

    merged_polygon = polygons[0]
    for poly in polygons[1:]:
        merged_polygon = merged_polygon.union(poly)

    total_area = merged_polygon.area

    return total_area

# Constant Variables
validation = r"C:\Users\cotyl\OneDrive\Desktop\CCIHP_icip\val_img processed"
targets_path = r"C:\Users\cotyl\OneDrive\Desktop\CCIHP_icip\val_seg processed head coordinates\val_seg processed coordinates.csv"
targets = csv_2_list(targets_path)
images = listdir(validation)
n = len(images)
weights = r"C:\Users\cotyl\OneDrive\Desktop\CCIHP_icip\CNNAnchorBoxes19Epochs.pth"
avg_ratio = 0
avg_IoU = 0
enclosed = 0
# Setting up the model
num_classes = 2
model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(load(weights, map_location=device("cpu")))
model.eval()
iterations = 1
#

for image, target in zip(images, targets):
    print(f'\nIteration {iterations}')
    iterations += 1
    path = join(validation, image)
    prediction = censor_img_test(path, model)

    if (target and prediction):
        actual_area = get_area(target)
        predicted_area = get_area(prediction)
        avg_ratio += actual_area / predicted_area

        target_polygons = [Polygon([(xi, yi), (xi, yf), (xf, yf), (xf, yi)]) for xi, yi, xf, yf in target]
        predicted_polygons = [Polygon([(xi, yi), (xi, yf), (xf, yf), (xf, yi)]) for xi, yi, xf, yf in prediction]
        merged_polygon = cascaded_union(predicted_polygons)

        censored = True
        for t_area in target_polygons:
            if not t_area.within(merged_polygon):
                censored = False
                break
        enclosed += censored

        intersection_area = sum([target.intersection(predicted).area for target in target_polygons for predicted in predicted_polygons])

        target_area = sum([target.area for target in target_polygons])
        predicted_area = sum([predicted.area for predicted in predicted_polygons])
        union_area = target_area + predicted_area - intersection_area

        IoU = intersection_area / union_area
        avg_IoU += IoU

    elif not target and not prediction:
        enclosed += 1
        avg_IoU += 1
        avg_ratio += 1

    elif not target:
        enclosed += 1

avg_ratio /= 5000
avg_IoU /= 5000
enclosed /= 5000

print(f'Average Ratio (Y/X): {avg_ratio}\nAverage IoU: {avg_IoU}\nAverage Rate of Strict Censoring: {enclosed}')

# For 15 epochs and with the filtering conditions l_limit=0.1, u_limit=0.5, l_ratio=0.1, u_ratio=5, score=0.995:
# Average Ratio (Y/X): 0.07821353863095591
# Average IoU: 0.07862995652056522

# For 19 epochs and with the filtering conditions l_limit=0.1, u_limit=0.5, l_ratio=0.1, u_ratio=5, score=0.995:
# Average Ratio (Y/X): 0.08787385315027263
# Average IoU: 0.09234938849012901 #########################

# For 19 epochs and with the filtering conditions l_limit=0.2, u_limit=0.5, l_ratio=0.2, u_ratio=4, score=0.995:
# Average Ratio (Y/X): 0.09718723855204958
# Average IoU: 0.08174292542415573

# For 19 epochs and with the filtering conditions l_limit=0.2, u_limit=0.5, l_ratio=0.2, u_ratio=4, score=0.999:
# Average Ratio (Y/X): 0.09740714215741912
# Average IoU: 0.08097548612791093

# For 19 epochs and with the filtering conditions l_limit=0.5, u_limit=0.5, l_ratio=0.2, u_ratio=4, score=0.999:
# Average Ratio (Y/X): 0.11296826527049357
# Average IoU: 0.07634205659948738

# For 19 epochs and with the filtering conditions l_limit=0.2, u_limit=0.5, l_ratio=0.33, u_ratio=4, score=0.99:
# Average Ratio (Y/X): 0.10954418820244698
# Average IoU: 0.09169371742169408

# For 19 epochs and with the filtering conditions l_limit=0.1, u_limit=0.3, l_ratio=0.33, u_ratio=4, score=0.95:
# Average Ratio (Y/X): 0.12822361113547323
# Average IoU: 0.08849830920261967

# For 19 epochs and with the filtering conditions l_limit=0.2, u_limit=0.4, l_ratio=0.33, u_ratio=4, score=0.99:
# Average Ratio (Y/X): 0.11401385564617025
# Average IoU: 0.07786791772677543

# For 19 epochs and with the filtering conditions l_limit=0.2, u_limit=0.4, l_ratio=0.33, u_ratio=4, score=0.99:
# Average Ratio (Y/X): 0.08806013062178857
# Average IoU: 0.09155584745148633
# Average Rate of Strict Censoring: 0.5274

# For 19 epochs and with the filtering conditions l_limit=0.2, u_limit=0.4, l_ratio=0.33, u_ratio=4, score=0.999:
#