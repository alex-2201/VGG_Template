
# Training in colab
https://gist.github.com/alex-2201/b3f1ea12abaeffadd4727081ee20172c

## Setup

In your drive, create:
- A folder named "secretKey" which contains your kaggle.json
- A folder named "VGG_Template" and then create a subfolder "checkpoint" inside to save the checkpoint

```console
from google.colab import drive
drive.mount('/content/gdrive')
!ln -s /content/gdrive/My\ Drive/ /mydrive
! pip install -q kaggle
! mkdir ~/.kaggle
! cp /mydrive/secretKey/kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle competitions download -c dogs-vs-cats-redux-kernels-edition
!unzip train
!unzip test
!mkdir data
!mv train data
!mv test data
!git clone https://github.com/alex-2201/VGG_Template.git
!mv data VGG_Template
cd VGG_Template/
```
## Train
```console
!python main.py
```

## Resume Train
```console
!python main.py --resume abc
```
