import gradio as gr
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from clip import CLIP
from models.tokenization import tokenizer
import gradio as gr
import yaml
from train import load_data
# 1. 加载 YAML 配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 为了方便调用，将配置项分别赋值
# 提示：YAML 中的列表 [80, 80] 在 Python 中是 list，有些模型要求 tuple，这里做一下转换
hp = config['hyperparameters']
vis_cfg = config['vision']
txt_cfg = config['text']
_, val_dataset, _ = load_data(data_path='../dataset/CLIP')
model_path='clip.pt'
# Load the model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hp = config['hyperparameters']
vis_cfg = hp['visual']
txt_cfg = hp['text']

retrieval_model = CLIP(
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
    retrieval=True,
).to(device)
retrieval_model.load_state_dict(torch.load("clip.pt", map_location=device))

# Function to process the query and return the top 30 images
def retrieve_images(query,device='cuda'):
    retrieval_model = retrieval_model(config, val_dataset, model_path='clip.pt')
    query_text, query_mask = tokenizer(query)
    query_text = query_text.unsqueeze(0).to(device)  # Add batch dimension
    query_mask = query_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        query_features = retrieval_model.text_encoder(query_text, mask=query_mask)
        query_features /= query_features.norm(dim=-1, keepdim=True)

    # Step 2: Encode all images in the dataset and store features
    image_features_list = []
    image_paths = []

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=5)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            features = retrieval_model.vision_encoder(images)
            features /= features.norm(dim=-1, keepdim=True)
            
            image_features_list.append(features)
            image_paths.extend(batch["id"])  # Assuming batch contains image paths or IDs

    # Concatenate all image features
    image_features = torch.cat(image_features_list, dim=0)

    # Step 3: Compute similarity using the CLIP model's logic
    similarities = (query_features @ image_features.T) * torch.exp(retrieval_model.temperature)
    similarities = similarities.softmax(dim=-1)

    # Retrieve top 30 matches
    top_values, top_indices = similarities.topk(30)

    # Step 4: Retrieve and display top N images
    images_to_display = []
    for value, index in zip(top_values[0], top_indices[0]):
        img_path = image_paths[index]
        img = Image.open(img_path).convert("RGB")
        images_to_display.append(np.array(img))

    return images_to_display

# Define the Gradio interface
def gradio_app(query):
    images = retrieve_images(query)
    return images




# Create Gradio Interface
with gr.Blocks() as interface:
    # Centered title
    gr.Markdown("<h1 style='text-align: center;'> 👒 Image Retrieval with CLIP -  👔👖 E-commerce Fashion 👚🥻</h1>")
    
    with gr.Row():
        # Textbox for query input
        query_input = gr.Textbox(placeholder="Enter your search query...", show_label=False, elem_id="custom-textbox")
        
        # Small submit button
        submit_btn = gr.Button("Search 🔍", elem_id="small-submit-btn")
    
    # Gallery output for displaying images
    gallery_output = gr.Gallery(label="Top 30 Matches").style(grid=[8], container=True)
    
    # Link the submit button to the function
    submit_btn.click(fn=gradio_app, inputs=query_input, outputs=gallery_output)

    # Custom CSS to make the submit button small and increase the font size in the textbox
    gr.HTML("""
    <style>
    #small-submit-btn {
        padding: 0.5rem 1rem;
        font-size: 0.8rem;
    }
    #custom-textbox input {
        font-size: 1.5rem;
    }
    </style>
    """)

# Launch the app
interface.launch()
