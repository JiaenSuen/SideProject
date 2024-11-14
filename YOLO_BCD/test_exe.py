import torch 
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm 
from torch.utils.data import DataLoader

from model import YOLOv1
from dataset import BCDataset
from utils import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)

from loss import YoloLoss

seed = 123
torch.manual_seed(seed)


#  -- Set --  #
Load_Model = True
Load_Model = False
Load_Model_File = "fit.pth.tar"

Learning_rate = 2e-5
Device = "cuda"
Batch_Size = 16
Weight_Decay = 0
Epoches = 100
Num_Worker = 0
PIN_memory = True

Img_Dir = "BCD/train/images"
Label_dir = "BCD/train/labels"
#  -- Set --  #

#  Transform  #
class Compose(object):
    def __init__(self,transforms) :
        self.transforms = transforms


    def __call__(self, img , bboxes ) :
        for t in self.transforms:
            img , bboxes = t(img) , bboxes
        return img, bboxes
transform = Compose([transforms.Resize((448,448)),transforms.ToTensor(),])

#  Transform  #

def Training (train_loader , model , optimizer , loss_fn ):
    loop = tqdm(train_loader , leave=True)
    mean_loss = []

    for batch_idx , (x,y)  in enumerate(loop):
        x , y = x.to(Device) , y.to(Device)
        out = model(x)
        loss = loss_fn(out ,y)
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

class_Num = 3
model = YOLOv1(split_size=7, num_boxes=2, num_classes=class_Num).to(Device)
optimizer = optim.Adam(
    model.parameters(), lr=Learning_rate, weight_decay=Weight_Decay
)

loss_fn = YoloLoss(C=class_Num)

if Load_Model:
    load_checkpoint(torch.load(Load_Model_File), model, optimizer)



train_dataset = BCDataset(img_dir=Img_Dir, label_dir=Label_dir,transform=transform, S=7, B=2, C=3)


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=Batch_Size,
    num_workers=Num_Worker,
    pin_memory=PIN_memory,
    shuffle=True,
    drop_last=True,
)




for x, y in train_loader:
    x = x.to(Device)
    for idx in range(8):
        bboxes = cellboxes_to_boxes(model(x))
        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

    import sys
    sys.exit()