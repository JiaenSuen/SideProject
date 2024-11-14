import torch 
import torch.nn as nn 


'''
    1. kernel_size
    2. out-channels
    3. stride 
    4. padding

'''

# 設定 YOLO 網路架構
architecture_config = [
    (7, 64 , 2 , 3),
    "M",
    (3, 192 , 1 , 1 ),
    "M",
    (1, 128 , 1 , 0),
    (3, 256 , 1 , 1),
    (1, 256 , 1 , 0),
    (3, 512 , 1 , 1),
    "M",
    [(1,256,1,0) , (3,512,1,1) , 4] ,
    (1,512 ,1,0),
    (3,1024,1,1),
    "M",
    [(1,512,1,0) , (3,1024,1,1) , 2],
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1),
]

# CNN 卷積層定義
class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNN_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


# YOLO v1 模型
class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        super(YOLOv1, self).__init__()
        self.in_channels = in_channels
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.darknet = self._create_conv_layers(architecture_config)
        self.fcs = self._create_fcs()

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [ 
                    CNN_block(in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3])
                ]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeat = x[2]

                for _ in range(num_repeat):
                    layers += [
                        CNN_block(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])
                    ]
                    layers += [
                        CNN_block(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self):
        S, B, C = self.split_size, self.num_boxes, self.num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )

# 測試模型
def test(split_size=7, num_boxes=2, num_classes=20):
    model = YOLOv1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)  # 預期輸出形狀為 (2, S*S*(C + B*5))

#test(num_classes=3)  # 例如測試3類別

 