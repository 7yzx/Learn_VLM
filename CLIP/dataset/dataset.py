from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image, ImageOps
from models.tokenization import tokenizer
import torch

class MyntraDataset(Dataset):
    def __init__(self, data_frame, captions, data_path ,target_size=28):
        self.data_frame = data_frame[data_frame['subCategory'].str.lower() != 'innerwear']
        
        self.target_size = target_size
        self.captions = captions
        self.data_path = data_path
        self.transform = T.Compose([T.ToTensor()])
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        while True:
            sample = self.data_frame.iloc[idx]
            img_path = os.path.join(self.data_path, f"{sample['id']}.jpg")
            
            try:
                image = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, IOError):
                idx = (idx + 1)% len(self.data_frame)
                continue
            # Resize the image to maintain aspect ratio
            image = self.resize_and_pad(image, self.target_size)
            image = self.transform(image)
            
            label = sample["subCategory"].lower()
            label = {"lips": "lipstick", "eyes": "eyelash", "nails": "nail polish"}.get(
                label, label
            )        
            label_idx = next(
                idx for idx, class_name in self.captions.items() if class_name == label
            )    
             # # Tokenize the caption using the tokenizer function
            cap, mask = tokenizer(self.captions[label_idx])
            # make sure mask is tensor
            mask = torch.tensor(mask)
            if len(mask.size()) == 1:
                mask = mask.unsqueeze(0)
            return {
                "image": image,
                "caption": cap,
                "mask": mask,
                "id": img_path,
            }
                
    def resize_and_pad(self, image, target_size):
        """
        保持比例缩放并填充到指定大小，如果有黑边横向填充为0，纵向填充为0
        
        
        :param self: Description
        :param image: Description
        :param target_size: Description
        """
        ori_w, orin_h = image.size
        aspect_ratio = ori_w / orin_h
        if aspect_ratio > 1:
            new_w = target_size
            new_h = int(target_size / aspect_ratio)
        else:
            new_h = target_size
            new_w = int(target_size * aspect_ratio)
        
        image = image.resize((new_w, new_h))
        pad_width = (target_size - new_w) // 2
        pad_height = (target_size - new_h) // 2
        padding = (
            pad_width, pad_height,
            target_size - new_w - pad_width, target_size - new_h - pad_height
        )
        image = ImageOps.expand(image, padding, fill=0) # 在四周填充黑色
        return image