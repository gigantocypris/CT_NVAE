# Multi-site COVID-Net CT Classification
Covid CT image classifier implemented with PyTorch. Forked from the original https://github.com/med-air/Contrastive-COVIDNet

## Usage

After cloning the repository, you'll need to download the datasets and put them in the `data/` folder. 

### Dataset

The authors utilized two publicly available COVID-19 CT datasets:

- [SARS-CoV-2 dataset](https://www.medrxiv.org/content/10.1101/2020.04.24.20078584v3)
- [COVID-CT dataset](http://arxiv.org/abs/2003.13865)

You can download their pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1JBp9RH9-yBEdtkNYDi6wWL79o62JD5Td/view?usp=sharing) and put it into the `data/` directory.

The path to the pre-processed datasets in CFS are:
- `/global/cfs/cdirs/m3562/users/lchien/Contrastive-COVIDNet/data/COVID-CT`
- `/global/cfs/cdirs/m3562/users/lchien/Contrastive-COVIDNet/data/SARS-Cov-2`

### Pretrained Model

You can also directly download their pretrained model from [Google Drive](https://drive.google.com/file/d/1ZwtxF4c_pvyv_uyE4Zx4_bNNHQx7Y_Ao/view?usp=sharing) and put into `saved/` directory for testing.

The path to the pretrained model in CFS is `/global/cfs/cdirs/m3562/users/lchien/Contrastive-COVIDNet/saved/best_checkpoint.pth`

### Training

Activate an interactive session.
```shell
salloc -N 1 --time=120 -C gpu -A m3562_g --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
```

Install libraries
```shell
pip install torch torchvision opencv-python pytorch_metric_learning
```

Start the job.
```shell
cd code
srun -n 1 python main.py --bna True --bnd True --cosine True --cont True
```

### Testing

```shell
cd code
python test.py /path/to/dataset/
```
