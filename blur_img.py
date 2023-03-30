from torch import from_numpy, load, device, no_grad
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from cv2 import imread, imshow, waitKey, destroyAllWindows, GaussianBlur, cvtColor, COLOR_BGR2RGB, imencode, resize, INTER_AREA, imdecode, IMREAD_COLOR, VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, VideoWriter, VideoWriter_fourcc, IMREAD_UNCHANGED
from os import listdir
from os.path import join, splitext
from numpy import array


def censor_img(file, l_limit=0.1, u_limit=0.5, l_ratio=0.1, u_ratio=4, score=0.995):
    """l_limit and u_limit are in percentage. This will filter out blur boxes that cover an area that is
    outside the accepted percentage of area covered. l_ratio and u_ratio is the direct ratio between the blur box's
    width over height. This will filter out boxes that are not within the acceptable range of ratios. score is also in
    percentage, and this will filter out boxes where the model's confidence that a box covers a face is not higher than
    the set threshold."""

    # Setting up the model
    num_classes = 2
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(load(r"path_to_weights_here", map_location=device("cpu")))
    model.eval()
    ########################################

    try:
        image = imdecode(file, IMREAD_COLOR)
        h, w = image.shape[0:2]
        width = w / 300
        height = h / 300

        # the following line of code makes a copy of the image and makes it a valid input for the model
        img = image.copy()
        img = resize(img, (300, 300), interpolation=INTER_AREA)
        img = cvtColor(img, COLOR_BGR2RGB)
        img = from_numpy(img.T)
        ##############################################

        # Model prediction
        with no_grad():
            output = model([img.float()])
        scores = output[0]['scores'].numpy()
        ############################


        # The following code removes blur boxes that are larger than a certain pixel amount and blur boxes where the
        # model is not confident that it covers a face
        selected_scores = scores >= score
        l_limit = l_limit * (h*w)
        u_limit = u_limit * (w*h)
        l_ratio = l_ratio
        u_ratio = u_ratio
        ##################################################################################

        # Filters out the boxes based on the set conditions above
        selected_predictions = {key: value[selected_scores] for key, value in output[0].items()}
        boxes = selected_predictions['boxes'].tolist()
        b = []
        for LIST in boxes:
            box = list(map(int, LIST))
            if (box[0] != box[2]) and (box[1] != box[3]):
                b.append(list(map(int, LIST)))
        b = [(int(box[0]*width), int(box[1]*height), int(box[2]*width), int(box[3]*height)) for box in b
             if u_limit >= (box[2] - box[0]) * (box[3] - box[1]) * width * height >= l_limit
             and l_ratio <= ((box[2] - box[0]) * width) / ((box[3] - box[1]) * height) <= u_ratio]
        ###########################################################

        # Applying the blur boxes onto the image with original dimensions
        for box in b:
            region = image[box[1]:box[3], box[0]:box[2]]
            blur = GaussianBlur(region, (31, 31), 0, 0)
            image[box[1]:box[3], box[0]:box[2]] = blur
        #############################################

        jpeg = imencode('.jpg', image)[1]
        return jpeg

    except:
        print("error")
        return "An error occurred while processing your file."


def censor_video2(file, path, l_limit=0.3, u_limit=0.7, l_ratio=0.15, u_ratio=5, score=0.95):
    """l_limit and u_limit are in percentage. This will filter out blur boxes that cover an area that is
    outside the accepted percentage of area covered. l_ratio and u_ratio is the direct ratio between the blur box's
    width over height. This will filter out boxes that are not within the acceptable range of ratios. score is also in
    percentage, and this will filter out boxes where the model's confidence that a box covers a face is not higher than
    the set threshold."""

    # Setting up the model
    global WIDTH, HEIGHT, wg, hg
    num_classes = 2
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(load(r"path_to_weights_here", map_location=device("cpu")))
    model.eval()
    ###############################################

    video = VideoCapture(path)
    wg = int(video.get(CAP_PROP_FRAME_WIDTH))
    hg = int(video.get(CAP_PROP_FRAME_HEIGHT))
    print(wg, hg)
    WIDTH = wg / 300
    HEIGHT = hg / 300
    fps = video.get(CAP_PROP_FPS)
    fourcc = VideoWriter_fourcc(*'vp09')
    output_folder = "./static"
    output_filename = splitext(file)[0] + '_processed.mp4'
    output_path = join(output_folder, output_filename)
    out = VideoWriter(output_path, fourcc, fps, (wg, hg))

    while video.isOpened():

        status, frame = video.read()
        if status:
            try:
                # the following lines of code converts the image into a usable format for the model
                image = frame.copy()
                img = resize(image, (300, 300), interpolation=INTER_AREA)
                img = cvtColor(img, COLOR_BGR2RGB)
                img = from_numpy(img.T)
                ############################################################################

                # Prediction of the model
                with no_grad():
                    output = model([img.float()])
                scores = output[0]['scores'].numpy()
                ############################################################################

                # The code below removes blur boxes that are larger/smaller than a certain percentage of pixel amount.
                # or ones where the model has a low confidence that it covers a face
                selected_scores = scores >= score
                l_limit_ = l_limit * (wg*hg)
                u_limit_ = u_limit * (wg*hg)
                l_ratio_ = l_ratio
                u_ratio_ = u_ratio
                ##################################################################################

                # The lines of code below filters out the blur boxes that do not follow the conditions above:
                selected_predictions = {key: value[selected_scores] for key, value in output[0].items()}
                boxes = selected_predictions['boxes'].tolist()
                b = []
                for LIST in boxes:
                    box = list(map(int, LIST))
                    if (box[0] != box[2]) and (box[1] != box[3]):
                        b.append(list(map(int, LIST)))
                b = [(int(box[0]*WIDTH), int(box[1]*HEIGHT), int(box[2]*WIDTH), int(box[3]*HEIGHT)) for box in b
                     if (u_limit_ >= (box[2] - box[0]) * (box[3] - box[1]) * WIDTH * HEIGHT >= l_limit_)
                     and (l_ratio_ <= ((box[2] - box[0]) * WIDTH) / ((box[3] - box[1]) * HEIGHT) <= u_ratio_)]
                ######################################################################################

                # The lines below add the blur boxes onto the video frame with original dimensions
                for box in b:
                    region = image[box[1]:box[3], box[0]:box[2]]
                    blur = GaussianBlur(region, (51, 51), 0, 0)
                    image[box[1]:box[3], box[0]:box[2]] = blur
                #######################################################

                out.write(image)

            except:
                print("error occurred")

        else:
            break

    out.release()
    video.release()


def censor_img_test(file, weights, l_limit=0.1, u_limit=0.5, l_ratio=0.1, u_ratio=5, score=0.995):
    """l_limit and u_limit are in percentage. This will filter out blur boxes that cover an area that is
    outside the accepted percentage of area covered. l_ratio and u_ratio is the direct ratio between the blur box's
    width over height. This will filter out boxes that are not within the acceptable range of ratios. score is also in
    percentage, and this will filter out boxes where the model's confidence that a box covers a face is not higher than
    the set threshold."""

    # Setting up the model
    num_classes = 2
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(load(weights, map_location=device("cpu")))
    model.eval()
    ########################

    try:
        image = imread(file)
        h, w = image.shape[0:2]
        width = w / 300
        height = h / 300

        # the following line of code converts the image from BGR to RGB channel format
        img = image.copy()
        img = resize(img, (300, 300), interpolation=INTER_AREA)
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
        v = []
        for LIST in boxes:
            box = list(map(int, LIST))
            if (box[0] != box[2]) and (box[1] != box[3]):
                b.append(list(map(int, LIST)))
                v.append(True)
            else:
                v.append(False)
        v = array(v)
        mask = array([((u_limit >= (box[2] - box[0]) * (box[3] - box[1]) * width * height >= l_limit)
                       and (l_ratio <= ((box[2] - box[0]) * width) / ((box[3] - box[1]) * height) <= u_ratio)) for box in b])
        b = [(int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)) for box in b
             if (u_limit >= (box[2] - box[0]) * (box[3] - box[1]) * width * height >= l_limit)
             and (l_ratio <= ((box[2] - box[0]) * width) / ((box[3] - box[1]) * height) <= u_ratio)]
        ###########################################

        # Information about the boxes that were not filtered out
        print(len(b), "boxes remain.")
        print(b)
        remaining_scores = scores[selected_scores][v][mask].tolist()
        print("Confidence Score:", remaining_scores)
        print("Average Score:", round(sum(remaining_scores)/len(remaining_scores), 8),
              f"(Lowest: {round(min(remaining_scores), 8)}, Highest: {round(max(remaining_scores), 8)})")
        percentages = [round((box[2]-box[0])*(box[3]-box[1])/(w*h), 3) for box in b]
        print("Percentage Occupied:", percentages)
        print("Average Percentage:", round(sum(percentages)/len(percentages), 4),
              f"(Lowest: {round(min(percentages), 4)}, Highest: {round(max(percentages), 4)})")
        ratios = [round((box[2]-box[0])/(box[3]-box[1]), 4) for box in b]
        print("Ratio of Boxes (W/H):", ratios)
        print("Average Ratio:", round(sum(ratios)/len(ratios), 4),
              f"(Lowest: {round(min(ratios), 4)}, Highest: {round(max(ratios), 4)})", "\n")
        ###################################################

        # Applies the blur boxes onto the image with the original dimensions
        for box in b:
            region = image[box[1]:box[3], box[0]:box[2]]
            blur = GaussianBlur(region, (31, 31), 0, 0)
            image[box[1]:box[3], box[0]:box[2]] = blur
        ##################################

        imshow("Edit", image)
        waitKey(0)
        destroyAllWindows()

    except:
        print("error")
        return "An error occurred while processing your file."


folder = r"path_to_image_folder_here"

if __name__ == "__main__":
    weights = r"path_to_weights_here"
    images = listdir(folder)
    for index in range(2000, 2100):
        censor_img_test(folder + rf"\{images[index]}", weights, l_limit=0.1, u_limit=0.45, l_ratio=0.75, u_ratio=4, score=0.995)
