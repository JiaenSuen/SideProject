
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import torch
from utils import*


BCD_list = ["Platelets",
            "RBC",
            "WBC"]



def show_data(dataset,start,end):
    for index in range(start,end+1):
        image, label_matrix = dataset[index]

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        ax = plt.gca()

        # 解析標註矩陣並畫出標註框
        S = dataset.S
        for i in range(S):
            for j in range(S):
                if label_matrix[i, j, 3] == 1:  # 確認物件存在
                    x_cell, y_cell, width_cell, height_cell = label_matrix[i, j, 4:8].tolist()
                    class_label = int(torch.argmax(label_matrix[i, j, :3]))

                    x = (j + x_cell) / S * image.width
                    y = (i + y_cell) / S * image.height
                    width = width_cell / S * image.width
                    height = height_cell / S * image.height

                    rect = patches.Rectangle(
                        (x - width / 2, y - height / 2), width, height,
                        linewidth=1, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)
                    plt.text(x, y, f"Class: {class_label}", color="white",
                            verticalalignment="bottom", bbox={"facecolor": "red", "alpha": 0.5})
        plt.axis("off")
        plt.show()





def show_predictions(dataset, model, start, end, device):
    for index in range(start, end + 1):
        image, label_matrix = dataset[index]
        image = image.to(device)

        # 進行模型推理，並將邊界框轉換為圖片中的坐標
        bboxes = cellboxes_to_boxes(model(image.unsqueeze(0)))  # 模型輸出的邊界框
        bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")

        # 顯示圖片
        plt.figure(figsize=(8, 8))
        
        # 如果需要使用 image.width 和 image.height，將 Tensor 轉回 PIL.Image
        image_pil = transforms.ToPILImage()(image.cpu())
        plt.imshow(image_pil)
        ax = plt.gca()

        # 解析標註矩陣並畫出標註框
        S = dataset.S
        height, width = image.shape[1], image.shape[2]  # 直接使用 Tensor 的形狀

        for i in range(S):
            for j in range(S):
                if label_matrix[i, j, 3] == 1:  # 如果標註中有物件
                    x_cell, y_cell, width_cell, height_cell = label_matrix[i, j, 4:8].tolist()
                    class_label = int(torch.argmax(label_matrix[i, j, :3]))
                    class_label_name = BCD_list[class_label]
                    # 計算框的位置和尺寸
                    x = (j + x_cell) / S * width  # 使用 Tensor 寬度
                    y = (i + y_cell) / S * height  # 使用 Tensor 高度
                    width = width_cell / S * width
                    height = height_cell / S * height

                    # 畫出標註框
                    rect = patches.Rectangle(
                        (x - width / 2, y - height / 2), width, height,
                        linewidth=1, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)
                    plt.text(x, y, f"Class: {class_label_name}", color="white",
                             verticalalignment="bottom", bbox={"facecolor": "red", "alpha": 0.5})

        # 打印 bboxes 結構
        #print(bboxes)

        # 根據 bboxes 的實際結構進行解包
        for bbox in bboxes:
            # 假設 bboxes 只有 6 個元素，進行解包
            x, y, w, h, conf, cls_pred = bbox
            x1 = (x - w / 2) * width
            y1 = (y - h / 2) * height
            box_width = w * width
            box_height = h * height

            rect = patches.Rectangle(
                (x1, y1), box_width, box_height,
                linewidth=2, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect)
            plt.text(x1, y1, f"Pred: {int(cls_pred)}", color="white",
                     verticalalignment="bottom", bbox={"facecolor": "green", "alpha": 0.5})

        plt.axis("off")
        plt.show()
