<div align="center">
<h1>Fin3R: Fine-tuning Feed-forward 3D Reconstruction Models via Monocular Knowledge Distillation </h1>

<a href="https://drive.google.com/file/d/1G3xkyV996VRDuCgzVCMBa4dz3iCftsKI/view?usp=sharing" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF">
</a>
<a href="https://arxiv.org/abs/2511.22429"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
<a href="https://visual-ai.github.io/fin3r/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>


**[Visual AI Lab, HKU](https://visailab.github.io/people.html)**; **[VIS, Baidu](https://vis.baidu.com/#/)**


[Weining Ren](https://github.com/rwn17), [Hongjun Wang](https://whj363636.github.io/), [Xiao Tan](https://tanxchong.github.io/), [Kai Han](https://www.kaihan.org/)
</div>

<h2 align="center">NeurIPS 2025</h2>


## Updates

- [Nov. 24, 2025] We release the inference code of Fin3R.


## Overview
Fin3R is a fine-tuning method designed to enhance the geometric accuracy and robustness of feed-forward 3D reconstruction models, while preserving their multi-view capability.


## Quick Start

First, clone this repository to your local machine, and install the dependencies (torch, torchvision, numpy, Pillow, and huggingface_hub). 
```bash
git clone git@github.com:Visual-AI/Fin3R.git 
cd Fin3R
pip install -r requirements.txt
```

You just need to apply lora weight by a single line of code.
```
model.apply_lora(lora_path = 'checkpoints/vggt_lora.pth')
```

Following VGGT demo, you can use it by: 

```python
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Add Fin3R Lora weight here!
model.apply_lora(lora_path = 'checkpoints/vggt_lora.pth')

# Load and preprocess example images (replace with your own image paths)
image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
```

The VGGT weights will be automatically downloaded from Hugging Face. If you encounter issues such as slow loading, you can manually download them [here](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt) and load, or:

```python
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.apply_lora(lora_path = 'checkpoints/vggt_lora.pth')
```

## Evaluation
Following [Pi3 evaluation code](https://github.com/yyfz/Pi3/tree/evaluation). The pointmap estimation results from two heads are as following:

| Method  | |  |  |**DTU**  | | |
|--------------|-----------------------------------|--------------------------------------|--------------------------------------|--------------------------------------|--------------------------------------|--------------------------------------| 
|              | **Acc. Mean** | **Acc. Med.**     | **Comp. Mean**    | **Comp. Med.**    | **N.C. Mean**     | **N.C. Med.**     |
| **VGGT cam+depth**| 1.298 | 0.754    | 1.964    | 1.033    | 0.666    | 0.752    |
| **Fin3R cam+depth**| 1.124 | 0.630    | 1.626    | 0.624    | 0.678    | 0.768    |
**VGGT pointmap**| 1.184 | 0.713    | 2.224    | 1.297    | 0.694    | 0.777    |
| **Fin3R pointmap**| 0.978  | 0.530 | 1.934 | 0.891 | 0.697| 0.785 |

| Method  | |  |  |**ETH3D**  | | |
|--------------|-----------------------------------|--------------------------------------|--------------------------------------|--------------------------------------|--------------------------------------|--------------------------------------| 
|              | **Acc. Mean** | **Acc. Med.**     | **Comp. Mean**    | **Comp. Med.**    | **N.C. Mean**     | **N.C. Med.**     |
| **VGGT cam+depth**| 0.285 | 0.195    | 0.338    | 0.213    | 0.834    | 0.931    |
| **Fin3R cam+depth**| 0.234 | 0.143    | 0.202    | 0.113    | 0.853    | 0.970    |
**VGGT pointmap**| 0.292 | 0.197    | 0.365    | 0.224    | 0.843    | 0.935    |
| **Fin3R pointmap**| 0.232  | 0.144 | 0.202 | 0.118 | 0.857 | 0.968 |


## Checkpoints

Checkpoints for DUSt3R, MASt3R, CUT3R and VGGT can be found at [Google Drive](https://drive.google.com/drive/folders/1dIy2-BYqYmXqY6_im-fS4UfOczXEG0WL?usp=sharing). We release the integration of DUSt3R [here](https://github.com/rwn17/fin3r_dust3r). You can also find the instructions at [issue#2](https://github.com/Visual-AI/Fin3R/issues/2).


## Interactive Demo

Based on the original demo provided by VGGT, we also provide multiple ways to visualize your 3D reconstructions. Before using these visualization tools, install the required dependencies:

```bash
pip install -r requirements_demo.txt
```

### Interactive 3D Visualization

**Please note:** VGGT typically reconstructs a scene in less than 1 second. However, visualizing 3D points may take tens of seconds due to third-party rendering, independent of VGGT's processing time. The visualization is slow especially when the number of images is large.


#### Gradio Web Interface

Our Gradio-based interface allows you to upload images/videos, run reconstruction, and interactively explore the 3D scene in your browser. You can launch this in your local machine or try it on [Hugging Face](https://huggingface.co/spaces/facebook/vggt).


```bash
python demo_gradio.py
```

<details>
<summary>Click to preview the Gradio interactive interface</summary>

![Gradio Web Interface Preview](https://jytime.github.io/data/vggt_hf_demo_screen.png)
</details>


#### Viser 3D Viewer

Run the following command to run reconstruction and visualize the point clouds in viser. Note this script requires a path to a folder containing images. It assumes only image files under the folder. You can set `--use_point_map` to use the point cloud from the point map branch, instead of the depth-based point cloud.

```bash
python demo_viser.py --image_folder path/to/your/images/folder
```

## Acknowledgements

Thanks to these great repositories: [PoseDiffusion](https://github.com/facebookresearch/PoseDiffusion), [VGGSfM](https://github.com/facebookresearch/vggsfm), [DINOv2](https://github.com/facebookresearch/dinov2), 
[DUSt3r](https://github.com/naver/dust3r), [MASt3R](https://github.com/naver/mast3r), [CUT3R](https://cut3r.github.io/),[Monst3r](https://github.com/Junyi42/monst3r), [VGGT](https://github.com/facebookresearch/vggt), [Moge](https://github.com/microsoft/moge), [PyTorch3D](https://github.com/facebookresearch/pytorch3d), [Sky Segmentation](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2),  and many other inspiring works in the community.

## Checklist
- [x] Release the DUSt3R integration and instructions
- [ ] Release the Evaluation code
- [ ] Release the training code


## License
All our model follows original license of each method. For example, for finetuned VGGT, see the [LICENSE](./LICENSE.txt) file for details.


## Citation

For any question, please contact [weining@connect.hku.hk](weining@connect.hku.hk). If you find this work useful, please cite

```bibtex
@inproceedings{ren2025fin3r,
  title={Fin3R: Fine-tuning Feed-forward 3D Reconstruction Models via Monocular Knowledge Distillation},
  author={Ren, Weining and Wang, Hongjun and Tan, Xiao and Han, Kai},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
