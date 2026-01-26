import json
# pretrain_dataset_path='/mnt/sevenT/zixiaoy/dataset/gongjy/minimind_dataset/sft_2048.jsonl'
pretrain_dataset_path='/mnt/sevenT/zixiaoy/dataset/gongjy/minimind_dataset/dpo.jsonl'



with open(pretrain_dataset_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        data = json.loads(line.strip())
        break
        
print(data.keys()) 
print(data)