
import torch
import torch.nn.functional as F


def calculate_DPO_loss(model_prefered_logprob, model_disprefered_logprob,
                       ref_prefered_logprob, ref_disprefered_logprob,
                       beta=0.5):
    
    # [关键点 1] 计算与参考模型的差距 (Relative Logprob)
    # 数学含义：log(π_theta / π_ref) = log(π_theta) - log(π_ref)
    # 这里的减法就是那个“除法”操作，用来防止模型跑偏。
    prefered_relative_logprob = model_prefered_logprob - ref_prefered_logprob
    disprefered_relative_logprob = model_disprefered_logprob - ref_disprefered_logprob

    # [关键点 2] 监控指标 (仅仅为了看，不参与梯度下降)
    # accuracy: 模型是否觉得“好答案”的分数比“坏答案”高？
    # margin: “好答案”比“坏答案”高出多少分？
    reward_accuracies = (prefered_relative_logprob > disprefered_relative_logprob).float().mean(dim=-1)
    reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean(dim=-1)

    # [关键点 3] 最终 Loss 计算
    # 公式：- log(sigmoid( beta * (好答案的分数 - 坏答案的分数) ))
    # 这一步就是让模型最大化(好-坏)的差值。
    loss = -F.logsigmoid(beta * (prefered_relative_logprob - disprefered_relative_logprob)).mean(dim=-1)

    return loss, reward_accuracies, reward_margins


def get_log_prob(logits, labels):
    """
    Docstring for get_log_prob
    Vocab Size (词表大小): 3 (词表里只有: 0:Apple, 1:Hello, 2:World)
    logits (模型原始输出): [[[10, 20, 5], [5, 10, 25]]] (形状: 1x2x3) # [B, Seq_len, Vocab_Size]
    labels (真实句子): [[1, 2]] (对应 "Hello World") # 形状: 1x2 [B, Seq_len]
    
    
    把“分数”变成“概率的对数”
    动作: Softmax 把分数变成 0~1 的概率，Log 把概率变成负数（对数概率）。
    """
    # 1. 把 Logits 转换成 Log Probabilities (归一化)
    log_probs = F.log_softmax(logits, dim=-1) # [B, Seq_len, Vocab_Size]
    
    # 2. 挑出我们关心的那个 token 的概率 (Gather)
    # 比如模型预测了 "apple", "banana"... 我们只要 label 里对应的那个词的概率。
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1).mean(-1)