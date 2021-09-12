from fastai.vision.all import *
from fastai.callback.mixup import *
from timm import create_model
import albumentations, timm
from fastai.metrics import accuracy, F1Score
from fastai.losses import BCEWithLogitsLossFlat, FocalLossFlat
import os
from fastai.distributed import *
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image
import torch
import cv2
import pandas as pd
import numpy as np

root = '/home/dsi/zuckerm1/BRCA_images/'
nfold = 2
device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else "cpu")
image_res=1024
batch_size = 4

data_transforms = {
    0: transforms.Compose([
        transforms.RandomResizedCrop(image_res), #224
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(90),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    1: transforms.Compose([
        transforms.CenterCrop(image_res),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class HpaDataset(Dataset):
    def __init__(self, is_valid, df):
        self.df = df
        self.df = self.df.loc[self.df.is_valid == is_valid]
        self.df.reset_index(inplace=True, drop=True)
        self.root = root
        self.transform = data_transforms[is_valid]
        self.is_valid = is_valid
        #self.vocab = [str(i) for i in range(0, 14)] #https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = np.array(Image.open(f'{self.root}/{self.df.ID[idx]}'))
        img = Image.fromarray(img)
        img = self.transform(img)
        label = np.array([float(x) for x in self.df.Label.iloc[idx].split('|')])
        return img, label

    def new_empty(self):
        return HpaDataset(False, self.df)


class AlbumentationsTransform(DisplayedTransform):
    split_idx,order=0,2
    def __init__(self, train_aug): store_attr()

    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)


def get_train_aug(): return albumentations.Compose([
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
            albumentations.CoarseDropout(p=0.5),
            albumentations.RandomContrast(p = 0.6)
])

def create_timm_body(arch:str, pretrained=True, cut=None):
    model = create_model(arch, pretrained=pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NamedError("cut must be either integer or function")

def get_model():
    body = create_body(resnet18, pretrained=True) #resnext50d_32x4d #resnet50 #resnet18
    nf = num_features_model(nn.Sequential(*body.children())) #* (2)
    head = create_head(nf, 14)
    model = nn.Sequential(body, head)
    apply_init(model[1], nn.init.kaiming_normal_)
    return model.cuda()


def test(model, testloader, critirion):
    y_hat_list = []
    model.eval()
    loss = 0
    with torch.no_grad():
        for x, y in testloader: 
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            y_hat_list.append(y_hat)
            test_loss = critirion(y, y_hat)
            loss += test_loss.item()
        loss = loss / len(trainloader)
    print('Test set: Average loss: {:.4f}'.format(loss))
    return y_hat_list


if __name__ == '__main__':
    df = pd.read_csv('updated_predictions_2.csv')
    train_ds = HpaDataset(False, df)
    valid_ds = HpaDataset(True, df)
    dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=batch_size, device=device)
    
    loss_func = MSELossFlat()
    model = get_model()
    learn = load_learner('baseline')
    
    preds = []
    for f in Path(root+'test/').iterdir():
        f= str(f)
        print(type(f))
        y_h,_,y_hat = learn.predict(f) #Image.open() #np.asarray(Image.open(f))
        y_h = y_h.numpy()
        y_hat = y_hat.numpy()
        preds.append((str(f), y_hat))
    col_names = ['image', 'pred']
    df = pd.DataFrame(preds, columns = col_names)
    df.to_csv("model_predictions.csv")
