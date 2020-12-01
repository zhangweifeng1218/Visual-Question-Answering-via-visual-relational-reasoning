import os
import sys
sys.path.insert(0, os.getcwd())
import h5py
import torch
import json
import torch.nn as nn
from torchvision.models.resnet import ResNet, resnet101
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from lib.util.transforms import Scale
from PIL import Image
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vqa_root = 'data/'

class ResNet (nn.Module):

    def __init__(self):

        super(ResNet, self).__init__()

        cnn = resnet101(pretrained=True)
        self.cnn = cnn
        self.features = nn.Sequential(cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool,
                                      cnn.layer1, cnn.layer2, cnn.layer3, cnn.layer4)

    def forward(self, x):
        """
        :param x: [ B, C, H, W ]
        """
        x = self.features(x)

        return x

    def freeze(self, layer="all"):
        if layer == "all":
            self.train(False)
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            layer = getattr(self, layer)
            layer.train(False)
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

transform = transforms.Compose([
    Scale([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

class VQAImg(Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        self.img_file_list = os.listdir(os.path.join(root, 'vqa', 'image', split))
        self.length = len(self.img_file_list)

    def __getitem__(self, index):
        img_file_name = self.img_file_list[index]
        short_name, extension = os.path.splitext(img_file_name)
        img_id_int = int(short_name.split('_')[-1])
        img_id = str(img_id_int)
        img = os.path.join(self.root, 'vqa', 'image', self.split, img_file_name)
        img = Image.open(img).convert('RGB')
        return img_id, transform(img)

    def __len__(self):
        return self.length

batch_size = 50

resnet = ResNet().to(device)
resnet.eval()

def create_dataset(split):
    dataset = VQAImg(vqa_root, split)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    total_img_num = len(dataset)
    print('found {} images in {} split ...'.format(total_img_num, split))

    f = h5py.File('data/vqa/spatial/{}_spatial.hdf5'.format(split), 'w', libver='latest')
    dset = f.create_dataset('data', (total_img_num, 2048, 7, 7), dtype='f4')

    info = {}

    idx = 0
    with torch.no_grad():
        for img_id, image in tqdm(dataloader):
            bs = image.size(0)
            image = image.to(device)
            features = resnet(image).detach().cpu().numpy()
            dset[idx:idx+bs] = features
            for i, id in enumerate(img_id):
                info[id] = idx + i
            idx += bs

    f.close()
    with open('data/vqa/spatial/{}_spatial_info.json'.format(split), 'w') as f:
        json.dump(info, f)
    print('{} image processed'.format(idx))

def extract_feature():
    create_dataset('train')
    create_dataset('val')
    create_dataset('test')
