# Multi-site COVID-Net CT Classification
Covid CT image classifier implemented with PyTorch. Forked from the original https://github.com/med-air/Contrastive-COVIDNet

## Usage

You'll need the following libraries
```shell
pip install torch torchvision opencv-python pytorch_metric_learning
```

### Dataset

The authors utilized two publicly available COVID-19 CT datasets:

- [SARS-CoV-2 dataset](https://www.medrxiv.org/content/10.1101/2020.04.24.20078584v3)
- [COVID-CT dataset](http://arxiv.org/abs/2003.13865)

The path to the pre-processed datasets in CFS are:
- `/global/cfs/cdirs/m3562/users/lchien/Contrastive-COVIDNet/data/COVID-CT`
- `/global/cfs/cdirs/m3562/users/lchien/Contrastive-COVIDNet/data/SARS-Cov-2`

You can also download their pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1JBp9RH9-yBEdtkNYDi6wWL79o62JD5Td/view?usp=sharing) and put it into the `data/` directory.

### Pretrained Model

The path to the pretrained model in CFS is `/global/cfs/cdirs/m3562/users/lchien/Contrastive-COVIDNet/saved/best_checkpoint.pth`

You can also directly download their pretrained model from [Google Drive](https://drive.google.com/file/d/1ZwtxF4c_pvyv_uyE4Zx4_bNNHQx7Y_Ao/view?usp=sharing) and put into `saved/` directory for testing.

### Training

Start the job.
```shell
cd code
srun -n 1 python main.py --bna True --bnd True --cosine True --cont True
```

### Testing the SARS-Cov-2 Dataset
To check what the `test_split.txt` file looks like, run:
```shell
cat /global/cfs/cdirs/m3562/users/lchien/Contrastive-COVIDNet/data/SARS-Cov-2/test_split.txt
```
If should look like this:

`COVID /global/cfs/cdirs/m3562/users/lchien/Contrastive-COVIDNet/data/SARS-Cov-2/COVID/Covid (409).png COVID-19`

If it instead has abbreviated paths like this: 
`COVID ../data/SARS-Cov-2/COVID/Covid (409).png COVID-19`

Then fix it:
```shell
cd code
python SARS_TEST_REWRITE.py
```

To actually test predictions:
```shell
cd code
python test.py /path/to/dataset/
```
