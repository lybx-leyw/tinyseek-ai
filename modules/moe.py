"""
尝试复现DeepSeekMoE架构
"""

import torch
import torch.nn as nn
import time

class FFN(nn.Module):
    def __init__(self,d_model,hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model,hidden)
        self.fc2 = nn.Linear(hidden,d_model)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MoE(nn.Module):
    def __init__(self,other_experts,shared_experts,d_model,hidden,device,keep,scale=0.02):
        super().__init__()
        self.other_experts = other_experts
        self.shared_experts = shared_experts
        self.d_model = d_model
        self.w_g = nn.Parameter(torch.randn(d_model, other_experts)*scale)
        self.w_n = nn.Parameter(torch.randn(d_model, other_experts)*scale)
        self.experts = nn.ModuleList(
            [
                FFN(d_model,hidden)
                for _ in range(other_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                FFN(d_model,hidden)
                for _ in range(shared_experts)
            ]
        )
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.keep = keep
        self.device = device
    
    def keep_topk(self,v,k):
        topk_values,topk_indices = torch.topk(v,k,dim=-1)
        result = -1e9*torch.ones_like(v) # v:[batch,seq_len,other_experts]
        result.scatter_(dim=-1,index=topk_indices,src=topk_values)
        return result
    
    def forward(self,x): # [batch,seq_len/time,d_model] #0.223s_test版0.336s
        batch_size,seq_len, _ = x.shape

        H_x = x@self.w_g 
        noise = torch.randn_like(H_x,device=self.device)*self.softplus(x@self.w_n)
        H_x = H_x + noise
        G_x = self.softmax(self.keep_topk(H_x,self.keep))
        # [batch,seq_len,other_experts]

        flat_G_x = G_x.view(-1,self.other_experts)
        flat_x = x.view(-1,self.d_model)

        output = torch.zeros(batch_size*seq_len,self.d_model,device=self.device)
        all_score = 0
        for expert_idx,expert in enumerate(self.experts):
            # 找出选择该专家的token
            mask = flat_G_x[:,expert_idx]!=0
            if mask.sum()==0:
                continue
            
            f_i = mask.sum()
            p_i = flat_G_x[:,expert_idx][mask].sum()
            score = self.other_experts*(f_i*p_i)/(self.keep*(flat_G_x.shape[0]**2))
            all_score += score.item()

            x_input = flat_x[mask]
            ffn_output = flat_G_x[mask][:,expert_idx:expert_idx+1]*expert(x_input)
            output[mask] += ffn_output
        
        for expert in self.shared_experts:
            output += expert(flat_x)

        output = output.reshape(batch_size,seq_len,self.d_model)

        return output,(all_score-1)
        
    
    """
    def forward_test(self,x): # [batch,seq_len/time,d_model]
        batch_size,seq_len, _ = x.shape

        H_x = x@self.w_g 
        noise = torch.randn_like(H_x)*self.softplus(x@self.w_n)
        H_x = H_x + noise
        G_x = self.softmax(self.keep_topk(H_x,self.keep))
        # [batch,seq_len,other_experts]

        flat_x = x.view(-1,self.d_model)

        all_expert_outputs = torch.stack([expert(flat_x) for expert in self.experts],dim=1)
        
        # 应用门控权重
        gating_weights = G_x.reshape(-1,self.other_experts)
        output = (all_expert_outputs*gating_weights.unsqueeze(-1)).sum(dim=1)
            
        for expert in self.shared_experts:
            output += expert(flat_x)

        output = output.reshape(batch_size,seq_len,self.d_model)
        return output
    """

if __name__ == '__main__':
    grained_coe = 4
    MoE_layer = MoE (
        other_experts = 6*grained_coe, 
        shared_experts = 2*grained_coe, 
        d_model = 1024,
        hidden = 4096//grained_coe, 
        device = torch.device('cpu'),
        keep = 2*grained_coe
    )
    all_time = 0
    for _ in range(30):
        moe_try_tensor = torch.randn(2,90,1024)
        start = time.perf_counter()
        output, score = MoE_layer(moe_try_tensor)
        end = time.perf_counter()
        all_time += end-start
    print(f"平均用时:{((all_time)/30):.2f}s")
    print(output.shape)
    print(score)
