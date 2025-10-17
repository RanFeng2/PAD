# PAD🗒️(Segmentation with RGB-SAR) 

> This is the official implementation of "**PAD: Phase-Amplitude Decoupling Fusion for Multi-Modal Land Cover Classification**"[[paper]](https://ieeexplore.ieee.org/document/11204689).


## Datasets
The following datasets are included for RGB-SAR semantic segmentation in this repo:

- **WHU-OPT-SAR**: A multi-modal dataset for land cover classification that includes RGB, NIR, and SAR images. The dataset is co-registered and covers **seven** land cover types: farmland, city, village, water, forest, road, and others.
- **DDHR-SK**: A dataset focusing on cloud-affected RGB-SAR imagery for land cover classification in Pohang City, South Korea. It includes **five** land cover types: buildings, roads, greenery, water, and farmland.


> All datasets and split files can be downloaded in [here](https://pan.baidu.com/s/1JkbsKOibCcQYfBR9sanG8g?pwd=36vk).

For all datasets, we performed necessary preprocessing, including:
- **extract the bands we need**
  - `WHUOPTSAR`: get RGB images from optical images(RGB+NIR)
- **cropped all images to the same area** (512x512 or 256x256)
  - `WHUOPTSAR`: 512x512
- **converted the labels to the range of** `[0, num_classes-1]`
  - `WHUOPTSAR`: -> [0, 7]. **Note:** `0` is background index in the original labels, which will be converted to `255`(ignore_index) in the dataset by `reduce_zero_label=True`.
- **split train/test dataset**
  - `WHUOPTSAR`: according to the ratio of 8:2

## How to Use
### Step 1: Download the Dataset
You can download all supported datasets by following the links in the previous section. The default data file structure is as follows:
```
<datasets_folder>
|-- <DatasetName1>
    |-- <CroppedImagesFiles>
        |-- <RGBFolder>
            |-- <name1>.<RGBFormat>
            |-- <name2>.<RGBFormat>
            ...
        |-- <SARFolder>
            |-- <name1>.<SARFormat>
            |-- <name2>.<SARFormat>
            ...
        |-- <LabelFolder>
            |-- <name1>.<LabelFormat>
            |-- <name2>.<LabelFormat>
            ...
    |-- <SplitedTxtFiles>
        |-- <train>.txt
        |-- <test>.txt
        |-- ...
|-- <DatasetName2>
|-- ...
```

e.g.:
```
|-- <WHUOPTSAR>
    |-- <all_512x512>
        |-- <opt>
            |-- NH49E001013_1.tif
            |-- NH49E001013_2.tif
            ...
        |-- <sar>
            |-- NH49E001013_1.tif
            |-- NH49E001013_2.tif
            ...
        |-- <lbl8>
            |-- NH49E001013_1.tif
            |-- NH49E001013_2.tif
            ...
    |-- <splited_files>
        |-- whuoptsar_train_(512, 512)_(512, 512).txt
        |-- whuoptsar_test_(512, 512)_(512, 512).txt
        |-- ...
```
which means `rgb`, `sar`, `label` files are stored in `opt`, `sar`, `lbl8` folders, respectively.

### Step 2: Create New Conda Environment


### Step 3: Test

### Step 4: Train
#### 1. Download Pretraned Weights：
All pre-trained weights can be downloaded [here](https://pan.baidu.com/s/1JkbsKOibCcQYfBR9sanG8g?pwd=36vk).

```
|-- <PAD>
    |-- <configs>
    |-- <models>
    |-- <pretrains>  # where to store pre-trained weights
    |-- <utils>
    |-- datasets_DDHR.py
    |-- datasets.py
    |-- main_DDHR.py
    |-- main.py
    |-- ...
```



## Results


## Citations
If these codes are helpful for your study, please cite:
```latex
@ARTICLE{zheng2025pad,
  author={Zheng, Huiling and Zhong, Xian and Liu, Bin and Xiao, Yi and Wen, Bihan and Li, Xiaofeng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={PAD: Phase-Amplitude Decoupling Fusion for Multi-Modal Land Cover Classification}, 
  doi={10.1109/TGRS.2025.3621902}
  }

```

