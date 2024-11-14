import torch
import os
from PIL import Image

class BCDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, S=7, B=2, C=3, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.B = B
        self.S = S
        self.C = C
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_filename = self.images[index]
        img_path = os.path.join(self.img_dir, img_filename)
        label_path = os.path.join(self.label_dir, f"{os.path.splitext(img_filename)[0]}.txt")

        # 開啟影像
        image = Image.open(img_path).convert("RGB")
        
        # 解析多點框標註
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                data = [float(x) for x in label.strip().split()]
                class_label = int(data[0])

                # 取得x_min, y_min, x_max, y_max來計算中心點與寬高
                x_min = min(data[1::2])
                y_min = min(data[2::2])
                x_max = max(data[1::2])
                y_max = max(data[2::2])

                x = (x_min + x_max) / 2
                y = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                boxes.append([class_label, x, y, width, height])
        
        boxes = torch.tensor(boxes)

        # 進行圖像增強
        if self.transform:
            image, boxes = self.transform(image, boxes)

        # 建立YOLO標註矩陣
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = width * self.S, height * self.S

            # 如果該網格尚未有物件標註
            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, self.C + 1: self.C + 5] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix