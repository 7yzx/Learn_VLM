from trl import SFTTrainer, SFTConfig
import torch
import wandb
import numpy as np

class CustomTrainerStage1(SFTTrainer):

    def set_feature_cache(self, feature_cache):
        """设置 VGGT 特征缓存管理器"""
        self.feature_cache = feature_cache

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        idx = inputs['idx']
        del inputs['idx']
        
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        predicted_ids = outputs.logits.argmax(dim=-1)
        decoded_text = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=False)
        predict_embeddings = outputs.hidden_states
        
        # all: 620
        image_out_mask = inputs["image_out_mask"]
        
        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

        # k_proj_weight = model.get_parameter('model.layers.17.self_attn.k_proj.weight')
        # x = [[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 151652]]
        # decoded_text = self.tokenizer.batch_decode(torch.tensor(x).to("cuda:0"), skip_special_tokens=False)
        
        ## the same
        input_embeddings = outputs.inputs_embeds
        
        mask = (inputs["input_ids"][0] == 151655).int()
        mask = mask.unsqueeze(0)
        image_embeddings = input_embeddings[mask.to(input_embeddings.device) != 0].contiguous()
        
        image_tokens = image_embeddings.shape[0]
        image_embed_dim = image_embeddings.shape[1]
        image_number = inputs["image_grid_thw"].shape[0]
        patch_size = int(image_tokens/image_number)
        image_embeddings = image_embeddings.view(image_number, patch_size, image_embed_dim).unsqueeze(0)
        
        feature_proj = model.projector_model(shift_predict_embeddings, image_embeddings)
        feature_proj_norm = feature_proj / feature_proj.norm(dim=-1, p=2, keepdim=True)
        
        # 使用特征缓存（异步预取 + LRU 缓存），避免每次从 NAS 同步读取
        if hasattr(self, 'feature_cache') and self.feature_cache is not None:
            feature_3d = self.feature_cache.get(idx[0], device=shift_predict_embeddings.device)
        else:
            # 回退到原始的 NAS 直接读取
            data = np.load('/mnt/sevenT/zixiaoy/code/Learn_VLM/Project/3DThinker/data/feature_vggt/' + str(idx[0]) + '/vggt.npz')
            feature_3d = data['feature'] # [1,N=4,P_3D = 1374,2048]
            if np.isnan(feature_3d).any() or np.isinf(feature_3d).any():
                print(f"!!! CRITICAL: GT Data contains NaN/Inf at idx {idx} !!!")
                feature_3d = np.nan_to_num(feature_3d)
            feature_3d = torch.tensor(feature_3d).to(device=shift_predict_embeddings.device, dtype=torch.float32)
        feature_3d = feature_3d.squeeze()
        
        # [检查点 2] Projector 输出是否有问题
        if torch.isnan(feature_proj).any() or torch.isinf(feature_proj).any():
            print(f"!!! CRITICAL: Projector output contains NaN/Inf at Step {self.state.global_step} !!!")
            print(f"Input stats: Min={shift_predict_embeddings.min()}, Max={shift_predict_embeddings.max()}")
            # 紧急补救：重置为0，防止训练崩溃
            feature_proj = torch.nan_to_num(feature_proj, nan=0.0, posinf=1.0, neginf=-1.0)

        diff = feature_proj - feature_3d.detach()
        squared_diff = diff ** 2
        
        if torch.isinf(squared_diff).any():
             print(f"!!! CRITICAL: Squared diff overflow at Step {self.state.global_step} !!!")
             squared_diff = torch.clamp(squared_diff, max=1e5) #再一次截断
             
        feature_sim = squared_diff.sum(dim=-1)
        sim_loss = feature_sim.mean()*0.0005
        
        if torch.isnan(sim_loss):
            print(f"!!! Warning: NaN detected in Sim Loss at Step {self.state.global_step} !!!")
        
        loss = 0.1 * ce_loss + sim_loss
        # loss = ce_loss + sim_loss
        wandb.log({
            "train/ce_loss": ce_loss.item(),
            "train/sim_loss": sim_loss.item(),
            "train/total_loss": loss.item(),
            "train/step": self.state.global_step,
            "train/epoch": self.state.epoch,
        })
        
        print(f"Step {self.state.global_step}: CE Loss: {ce_loss.item():.4f}, Sim Loss: {sim_loss.item():.4f}, Total Loss: {loss.item():.4f}")
        
        # 每 100 步打印一次缓存统计
        if hasattr(self, 'feature_cache') and self.feature_cache is not None and self.state.global_step % 100 == 0:
            self.feature_cache.log_stats()
        
        return (loss, outputs) if return_outputs else loss

class CustomTrainerStage2(SFTTrainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        loss = ce_loss
        wandb.log({
            "train/ce_loss": ce_loss.item(),
            "train/total_loss": loss.item(),
            "train/step": self.state.global_step,
            "train/epoch": self.state.epoch,
        })
        print(f"Step {self.state.global_step}: CE Loss: {ce_loss.item():.4f}")
        return (loss, outputs) if return_outputs else loss