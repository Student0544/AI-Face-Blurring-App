from torch import from_numpy, load, device, no_grad
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from cv2 import imread, imshow, waitKey, destroyAllWindows, GaussianBlur, cvtColor, COLOR_BGR2RGB, imencode, resize, INTER_AREA, imdecode, IMREAD_COLOR, VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, VideoWriter, VideoWriter_fourcc, IMREAD_UNCHANGED
from os import listdir
import os



def censor_img(file):
    num_classes = 2
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(load(r"C:\Users\cotyl\OneDrive\Desktop\CCIHP_icip\CNNTotal.pth", map_location=device("cpu")))
    model.eval()

    try:
        # image = imdecode(file, IMREAD_COLOR)
        image = imread(file)
        print("done")
        image = resize(image, (300, 300), interpolation=INTER_AREA)

        img = image.copy()
        # the following line of code converts the image from BGR to RGB channel format
        img = cvtColor(img, COLOR_BGR2RGB)
        img = from_numpy(img.T)

        with no_grad():
            output = model([img.float()])
        scores = output[0]['scores'].numpy()
        selected_scores = scores >= 0.5

        selected_predictions = {key: value[selected_scores] for key, value in output[0].items()}
        boxes = selected_predictions['boxes'].tolist()
        # The following code removes blur boxes that are larger than a certain pixel amount
        l_limit = 0
        u_limit = 20000
        l_ratio = 0.1
        u_ratio = 5
        ##################################################################################

        b = []
        for l in boxes:
            if len(set(list(map(int, l)))) == 4:
                b.append(list(map(int, l)))

        b = [box for box in b if u_limit >= (box[2] - box[0]) * (box[3] - box[1]) >= l_limit
            and l_ratio <= (box[2] - box[0]) / (box[3] - box[1]) <= u_ratio]
        print(b)
        for box in b:
            region = image[box[1]:box[3], box[0]:box[2]]
            blur = GaussianBlur(region, (31, 31), 0, 0)
            image[box[1]:box[3], box[0]:box[2]] = blur

        imshow("First Edit", image)
        waitKey(0)
        destroyAllWindows()
        # print("done")
        # jpeg = imencode('.jpg', image)[1]
        # return jpeg

    except:
        print("error")
        return "An error occurred while processing your file."


def censor_video(file):
    num_classes = 2
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(load(r"C:\Users\cotyl\OneDrive\Desktop\CCIHP_icip\CNNTotal.pth", map_location=device("cpu")))
    model.eval()

    # file = imdecode(file, IMREAD_UNCHANGED)
    video = VideoCapture(file)
    fps = video.get(CAP_PROP_FPS)
    fourcc = VideoWriter_fourcc(*'mp4v')
    out = VideoWriter(r'output_video2.mp4', fourcc, fps, (300, 300))

    while video.isOpened():
        status, frame = video.read()
        print(status)
        if status:
            try:
                image = resize(frame, (300, 300), interpolation=INTER_AREA)

                img = image.copy()
                # the following line of code converts the image from BGR to RGB channel format
                img = cvtColor(img, COLOR_BGR2RGB)
                img = from_numpy(img.T)

                with no_grad():
                    output = model([img.float()])
                scores = output[0]['scores'].numpy()

                ################################
                selected_scores = scores >= 0.9
                #################################

                selected_predictions = {key: value[selected_scores] for key, value in output[0].items()}
                boxes = selected_predictions['boxes'].tolist()

                # The following code removes blur boxes that are larger than a certain pixel amount
                l_limit = 1000
                u_limit = 40000
                l_ratio = 0.25
                u_ratio = 5
                ##################################################################################

                boxes = [box for box in boxes if u_limit >= (box[2] - box[0]) * (box[3] - box[1]) >= l_limit
                         and l_ratio <= (box[2] - box[0]) / (box[3] - box[1]) <= u_ratio]
                b = []
                for l in boxes:
                    if len(set(list(map(int, l)))) == 4:
                        b.append(list(map(int, l)))
                b = [box for box in b if u_limit >= (box[2] - box[0]) * (box[3] - box[1]) >= l_limit
                         and l_ratio <= (box[2] - box[0]) / (box[3] - box[1]) <= u_ratio]

                for box in b:
                    region = image[box[1]:box[3], box[0]:box[2]]
                    blur = GaussianBlur(region, (31, 31), 0, 0)
                    image[box[1]:box[3], box[0]:box[2]] = blur

                out.write(image)

            except:
                print("error occurred")

        else:
            break
    out.release()
    video.release()


def censor_video2(file, path):
    num_classes = 2
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(load(r"C:\Users\cotyl\OneDrive\Desktop\CCIHP_icip\CNNTotal.pth", map_location=device("cpu")))
    model.eval()

    video = VideoCapture(path)
    fps = video.get(CAP_PROP_FPS)
    fourcc = VideoWriter_fourcc(*'H264')
    print("hello")
    output_folder = "./static"
    output_filename = os.path.splitext(file)[0] + '_processed.mp4'
    output_path = os.path.join(output_folder, output_filename)
    out = VideoWriter(output_path, fourcc, fps, (300, 300))

    # out = VideoWriter(r"output_video.mp4", fourcc, fps, (300, 300))

    while video.isOpened():
        status, frame = video.read()
        print(status)
        if status:
            try:
                image = resize(frame, (300, 300), interpolation=INTER_AREA)

                img = image.copy()
                # the following line of code converts the image from BGR to RGB channel format
                img = cvtColor(img, COLOR_BGR2RGB)
                img = from_numpy(img.T)

                with no_grad():
                    output = model([img.float()])
                scores = output[0]['scores'].numpy()

                ################################
                selected_scores = scores >= 0.9
                #################################

                selected_predictions = {key: value[selected_scores] for key, value in output[0].items()}
                boxes = selected_predictions['boxes'].tolist()

                # The following code removes blur boxes that are larger than a certain pixel amount
                l_limit = 1000
                u_limit = 40000
                l_ratio = 0.25
                u_ratio = 5
                ##################################################################################


                b = []
                for l in boxes:
                    if len(set(list(map(int, l)))) == 4:
                        b.append(list(map(int, l)))
                b = [box for box in b if u_limit >= (box[2] - box[0]) * (box[3] - box[1]) >= l_limit
                    and l_ratio <= (box[2] - box[0]) / (box[3] - box[1]) <= u_ratio]

                for box in b:
                    region = image[box[1]:box[3], box[0]:box[2]]
                    blur = GaussianBlur(region, (31, 31), 0, 0)
                    image[box[1]:box[3], box[0]:box[2]] = blur

                out.write(image)

            except:
                print("error occurred")

        else:
            break

    out.release()
    video.release()
    print(output_filename)
    # return output_filename


    # with open("output_buffer.mp4", "rb") as f:
    #     video_bytes = f.read()
    # encoded_bytes = imencode('.mp4', frombuffer(video_bytes, uint8))[1].tobytes()
    # return encoded_bytes

# images = listdir(r"C:\Users\cotyl\OneDrive\Desktop\CCIHP_icip\train_img processed")
#
# for index in range(60, 100):
#     censor_img(rf"C:\Users\cotyl\OneDrive\Desktop\CCIHP_icip\train_img processed\{images[index]}")

# censor_video2(r"C:\Users\cotyl\OneDrive\Desktop\CCIHP_icip\test.mp4")