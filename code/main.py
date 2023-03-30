from google.colab import drive
drive.mount('/content/drive/')

! cp "/content/drive/MyDrive/CNN Data/train_img processed.zip" "/content"
! unzip "/content/train_img processed.zip"
! gdown --id 1bJxgF5NfBmz1WGVvYNvsEp59udSs78j5

import csv
from glob import glob
import cv2
from torch import from_numpy, tensor, cuda, optim, device, save, load
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = device("cuda:0" if cuda.is_available() else "cpu")
print(device)
cuda.empty_cache()

training_folder = "/content/train_img processed/"
training_path = "/content/train_seg processed coordinates3.csv"


def img_2_pixel(folder):
    """IMPORTANT: The images will come out in BGR. Please precise whether you'd like to use float32 or float64 in the
    calculations. This returns a numpy array that turns all images in the folder into its pixel values. Note that it will
    yield a 3D array as there are 3 color channels."""

    images = list(glob(f"{folder}*.jpg"))  # The image paths are all here
    images = [cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB) for image in images]

    return images


def csv_2_list(csv_file, device, img_s=300):
    """Returns a numpy array where each element is an array containing the coordinates of the boundary boxes in a vector.
    Each 4 coordinate corresponds to one box, hence there's n/4 faces per vector of n elements. Please precise whether
    you want int size 64, 32, 16 or 8."""
    with open(csv_file, "r", newline="") as f:

        file = csv.reader(f)
        arr = []

        g = lambda y: float(y) / img_s
        for line in file:

            if line == ["-"]:
                arr.append({'boxes': tensor([[0, 0, 1, 1]]), 'labels': tensor([0])})

            else:
                line = list(map(g, line))
                box = [line[4 * bracket:4 * bracket + 4] for bracket in range(len(line) // 4)]

                label = [1] * len(box)

                arr.append({'boxes': tensor(box), 'labels': tensor(label)})

            # line = list(map(g, line))
            # box = [line[4 * bracket:4 * bracket + 4] for bracket in range(len(line) // 4)]

            # dic = {True: 1, False: 0}
            # label = [dic[l != [0.0, 0.0, 0.01, 0.01]] for l in box]

            # arr.append({'boxes': tensor(box), 'labels': tensor(label)})

    return arr


class FaceDataset(Dataset):
    # Load the data
    def __init__(self, data_folder, expected_value_file):
        self.x = img_2_pixel(data_folder)
        print(len(self.x))

        # If using padding, use the line below:
        self.y = csv_2_list(expected_value_file, device)

        self.n_samples = len(self.x)

    # Return size of dataset
    def __len__(self):
        return self.n_samples

    # Allow index search
    def __getitem__(self, index):
        """Returns two values separately, so you'll have to unpack it with two designated variables"""
        # Writing self[index] will return these two values
        return from_numpy(self.x[index].T).float(), self.y[index]


width = 300
height = 300
batch_size = 32
dataset = FaceDataset(training_folder, training_path)


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

num_classes = 2
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
print("loading")
device = "cuda:0"
print("done")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.01,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# model.load_state_dict(load("/content/drive/MyDrive/CNN Data/CNNTotal.pth", map_location = 'cpu'))
num_epochs = 10

if cuda.is_available():
    model.cuda()
model.train()

print("Starting iterations")

for epoch in range(num_epochs):
    for images, targets in dataloader:
        cuda.empty_cache()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        loss_dict = model(images, targets)
        print(epoch+1, f"/{num_epochs}:", loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()

        optimizer.zero_grad()
        optimizer.step()
        lr_scheduler.step()

        # images = list(image.to("cpu") for image in images)
        # targets = [{k: v.to("cpu") for k, v in t.items()} for t in targets]

print("done")

# Save the trained model
save(model.state_dict(), r'/content/drive/MyDrive/CNN Data/CNNFinal2.pth')