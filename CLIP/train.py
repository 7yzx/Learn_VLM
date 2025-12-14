from clip import CLIP
import torch
import yaml  # 需要安装 pyyaml
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.dataset import MyntraDataset
import os
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import kagglehub
import argparse
from models.tokenization import tokenizer
# 添加代理
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

# 1. 加载 YAML 配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 为了方便调用，将配置项分别赋值
# 提示：YAML 中的列表 [80, 80] 在 Python 中是 list，有些模型要求 tuple，这里做一下转换
hp = config['hyperparameters']
vis_cfg = config['vision']
txt_cfg = config['text']

def download_data(data_path):
    import shutil
    path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
    shutil.move(path, data_path)
    
def load_data(data_path):
    # Load the dataset
    csv_path = os.path.join(data_path, 'styles.csv')
    df = pd.read_csv(csv_path, usecols=['id',  'subCategory'])
    
    unique, counts = np.unique(df["subCategory"].tolist(), return_counts = True)
    print(f"Classes: {unique}: {counts}")
    
    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.10, random_state=42)
    
    # Print the sizes of the datasets
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
    class_names = df['subCategory'].unique()
    class_names = [str(name).lower() for name in class_names]
    
    # Replace in-place 把其中指代不明的替换为明确的
    for i, name in enumerate(class_names):
        if name == "lips":
            class_names[i] = "lipstick"
        elif name == "eyes":
            class_names[i] = "eyelash"
        elif name == "nails":
            class_names[i] = "nail polish"
    
    captions = {idx: class_name for idx, class_name in enumerate(class_names)}
    
    for idx, caption in captions.items():
        print(f"{idx}: {caption}\n") 
    
    train_dataset = MyntraDataset(data_frame=train_df, captions=captions, data_path=data_path, target_size=80)
    val_dataset = MyntraDataset(data_frame=val_df, captions=captions, data_path=data_path, target_size=80)
    test_dataset = MyntraDataset(data_frame=val_df, captions=captions, data_path=data_path, target_size=224)
    
    print(f"Number of Samples in Train: {len(train_dataset)}")
    print(f"Number of Samples in Val: {len(val_dataset)}")
    
    # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=hp['batch_size'], num_workers=4)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=hp['batch_size'], num_workers=4)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=hp['batch_size'], num_workers=4)
    # print(f"Batch size and data shapes: {len(next(iter(train_loader)))}")  # (img_tensor,label_tensor)
    return train_dataset, val_loader, test_loader


def train(data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CLIP(
        emb_dim=hp['emb_dim'],
        # 视觉部分
        vit_layers=vis_cfg['layers'],
        vit_d_model=vis_cfg['d_model'],
        img_size=vis_cfg['img_size'],      # 转换 list -> tuple
        patch_size=vis_cfg['patch_size'],  # 转换 list -> tuple
        n_channels=vis_cfg['n_channels'],
        vit_heads=vis_cfg['heads'],
        # 文本部分
        vocab_size=txt_cfg['vocab_size'],
        max_seq_length=txt_cfg['max_seq_length'],
        text_heads=txt_cfg['heads'],
        text_layers=txt_cfg['layers'],
        text_d_model=txt_cfg['d_model'],
        retrieval=False,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=hp['lr'])
    
    total_params = 0 
    total_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
    
    print(f'Total parameters: {total_params}')
    
    print("Loading data...")
    train_loader, val_loader, test_loader = load_data(data_path)
    print("Loading data Done...")
    best_loss = np.inf
    for epoch in range(hp['epochs']):
        epoch_loss = 0
        with tqdm(enumerate(train_loader,0), total=len(train_loader), desc=f"Epoch {epoch + 1}/{hp['epochs']}") as t:
            for i, data in t:
                img, cap, mask = (
                    data["image"].to(device),
                    data["caption"].to(device),
                    data["mask"].to(device),
                )
                optimizer.zero_grad()
                loss = model(img, cap, mask)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                t.set_postfix(loss=loss.item())
                epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{hp["epochs"]}, Loss: {avg_loss}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'clip.pth')
            print('Best model saved with loss:', best_loss)
        
    test(val_loader, 'clip.pth', data_path)



def test(val_dataset, model_path, data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CLIP(
        emb_dim=hp['emb_dim'],
        # 视觉部分
        vit_layers=vis_cfg['layers'],
        vit_d_model=vis_cfg['d_model'],
        img_size=vis_cfg['img_size'],      # 转换 list -> tuple
        patch_size=vis_cfg['patch_size'],  # 转换 list -> tuple
        n_channels=vis_cfg['n_channels'],
        vit_heads=vis_cfg['heads'],
        # 文本部分
        vocab_size=txt_cfg['vocab_size'],
        max_seq_length=txt_cfg['max_seq_length'],
        text_heads=txt_cfg['heads'],
        text_layers=txt_cfg['layers'],
        text_d_model=txt_cfg['d_model'],
        retrieval=False,
    ).to(device)
    model.load_state_dict(torch.load(model_path), map_location=device)
    
    text = torch.stack([tokenizer(x)[0] for x in val_dataset.captions.values()]).to(device)
    mask = torch.stack([tokenizer(x)[1] for x in val_dataset.captions.values()]).to(device)
    mask = (mask.repeat(1, len(mask[0])).reshape(len(mask), len(mask[0]), len(mask[0]))).to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for data in val_dataset:
            images, labels = data["image"].to(device), data["label"].to(device)
            image_features = model.vision_encoder(images)
            text_features = model.text_encoder(text, mask)
            image_features /= image_features.norm(dim=-1 ,keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
            _, indices = torch.max(similarity, dim=-1)
            
            pred = torch.stack(
                [tokenizer(val_dataset.captions[int(i)])[0] for i in indices]
            )
            correct += int(sum(torch.sum(pred == labels, dim=1) // len(pred[0])))
            total += len(labels)
    print(f"Accuracy: {100 * correct / total}%")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP model")
    parser.add_argument("--download_data", action="store_true", help="Download dataset from Kaggle")
    parser.add_argument("--data_path", type=str, default="../dataset/CLIP", help="Path to the dataset")
    args = parser.parse_args()
    
    
    
    if args.download_data:
        download_data(args.data_path)
    train(data_path=args.data_path)