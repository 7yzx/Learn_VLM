#wget -c -O ChartQA.zip "https://huggingface.co/datasets/ReFocus/ReFocus_Data/resolve/main/images/ChartQA.zip?download=true"
python download_dataset.py
cd /mnt/yzx/zixiao/datasets/ReFocus_Data/images
unzip ChartQA.zip
unzip train_chartqa_vcot.zip

# python preprocess.py
