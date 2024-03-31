import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import sampler
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from pathlib import Path
from PIL import Image

    
class ChunkSampler(sampler.Sampler):
  
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples
    

img_dir = Path("/content/drive/My Drive/Dataset/")


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_dir_list = list(img_dir.glob("*"))
        self.transform = transform

    def __len__(self):
        return len(self.img_dir_list)

    def __getitem__(self, idx):
        img_path = self.img_dir_list[idx]
        image = Image.open(img_path)  

        desired_size = (256, 256)
        image = image.resize(desired_size)

        if self.transform:
            image = self.transform(image)

        return image
    
transform = transforms.Compose([transforms.ToTensor()])

    
dataset = CustomImageDataset(img_dir, transform=transform)
    

def get_dataloader_custom(batch_size, num_workers=0, train_transforms=None):

    if train_transforms is None:
        train_transforms = ToTensor 

    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=False)
    return train_loader
