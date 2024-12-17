# DS-473 Project 👁
Everything was run locally using PyTorch.

## Pre-trained models
### Yolo
- Directly taken from Ultralytics python package

### R-CNN
- Taken from torchvision
  - ResNet50 (heavy)
  - MobileNet (light)

## Dataset structure

### Old structure
```txt
datasets/
├── anno/
│   └── instances_val2017.json
└── images/
    └── val2017/
        ├── 000001.jpg
        ├── 000002.jpg
        ├── 000003.jpg
        └── ...
```


### New structure
For this I had to use `ultralytics.data.converter.convert_coco()`. This output the labels in a different directory. I pulled them into the original dataset directory.
```txt
datasets/
├── anno/
│   └── instances_val2017.json   # coco-style, json labels
├── images/
│   └── val2017/
│       ├── 000001.jpg
│       ├── 000002.jpg
│       ├── 000003.jpg
│       └── ...
└── labels/
    └── val2017/
        ├── 000001.txt            # Yolo-style, txt labels
        ├── 000002.txt
        ├── 000003.txt
        └── ...
```

