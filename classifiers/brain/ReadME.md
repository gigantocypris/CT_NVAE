# 2D CNN Classifier
The original author's repository is found at https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection

## Pretrained models
Download the following pretrained models and unzip them in `/src/` (I used GLOBUS to accomplish this).
- seresnext101_256*256 [\[seresnext101\]](https://drive.google.com/open?id=18Py5eW1E4hSbTT6658IAjQjJGS28grdx)
- densenet169_256*256 [\[densenet169\]](https://drive.google.com/open?id=1vCsX12pMZxBmuGGNVnjFFiZ-5u5vD-h6)
- densenet121_512*512 [\[densenet121\]](https://drive.google.com/open?id=1o0ok-6I2hY1ygSWdZOKmSD84FsEpgDaa)

## Preprocessing
![image](https://github.com/SeuTao/RSNA2019_1st_place_solution/blob/master/docs/preprocessing.png)

Download the CSV files and upzip them `/data/` (I used GLOBUS to accomplish this).

download [\[data.zip\]](https://drive.google.com/open?id=1buISR_b3HQDU4KeNc_DmvKTYJ1gvj5-3)

## Dataset
- The dataset is currently stored in CFS: `/global/cfs/cdirs/m3562/users/hkim/brain_data/raw`. 
- If you need to download the dataset again, it is from Kaggle: [\[RSNA Intracranial Hemorrhage Detection\]](https://drive.google.com/open?id=18Py5eW1E4hSbTT6658IAjQjJGS28grdx) 
- To download from Kaggle:
    1. make sure NERSC has your API credentials: [\[Kaggle API\]](https://github.com/Kaggle/kaggle-api)
    2. Then "join" the _RSNA Intracranial Hemorrhage Detection_ competition on Kaggle.
    3. In your command line, install Kaggle: `pip install kaggle`
    4. Download the dataset with: `kaggle competitions download -c rsna-intracranial-hemorrhage-detection`
    5. Unzip it in `/data/`

## Convert dcm to png
I have already done this step. You can move them from the path below to your `/src/` folder: 
```
/global/cfs/cdirs/m3562/users/lchien/kaggle/2DNet/src/train_png
/global/cfs/cdirs/m3562/users/lchien/kaggle/2DNet/src/test_png
```
Alternatively, if you need to do this again:
```
cd src
python3 prepare_data.py -dcm_path $PATH_TO_TRAIN_IMAGES -png_path train_png
python3 prepare_data.py -dcm_path $PATH_TO_TEST_IMAGES -png_path test_png
```

## Training
To train models, use the following commands:
```
python3 train_model.py -backbone DenseNet121_change_avg -img_size 256 -tbs 256 -vbs 128 -save_path DenseNet121_change_avg_256
python3 train_model.py -backbone DenseNet169_change_avg -img_size 256 -tbs 256 -vbs 128 -save_path DenseNet169_change_avg_256
python3 train_model.py -backbone se_resnext101_32x4d -img_size 256 -tbs 80 -vbs 40 -save_path se_resnext101_32x4d_256
```

## Testing
To test models, use the following commands:
```
python3 predict.py -backbone DenseNet121_change_avg -img_size 256 -tbs 4 -vbs 4 -spth DenseNet121_change_avg_512
python3 predict.py -backbone DenseNet169_change_avg -img_size 256 -tbs 4 -vbs 4 -spth DenseNet169_change_avg_256
python3 predict.py -backbone se_resnext101_32x4d -img_size 256 -tbs 4 -vbs 4 -spth se_resnext101_32x4d_256
```

## TODO:
1. Create a .gitignore
