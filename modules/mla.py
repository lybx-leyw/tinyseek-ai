"""
尝试复现DeepSeekMLA架构
"""

import torch
import torch.nn as nn
import math
import time

class MLA(nn.Module):
    def __init__(self,d_model,d_c,d_r,n_head,device,ro_theta=10000.0):
        super().__init__()
        self.device = device
        self.ro_theta = ro_theta
        self.d_r = d_r
        self.d_model = d_model
        self.n_head = n_head

        self.w_dq = nn.Linear(d_model,d_c)
        self.w_dkv = nn.Linear(d_model,d_c)

        self.w_qr = nn.Linear(d_c,d_r)
        self.w_kr = nn.Linear(d_model,d_r)

        self.w_uq = nn.Linear(d_c,d_model)
        self.w_uk = nn.Linear(d_c,d_model)
        self.w_uv = nn.Linear(d_c,d_model)

        self.w_o = nn.Linear(d_model,d_model)
        self.softmax = nn.Softmax(dim=-1)
    
    def rope(self,x): # [batch,seq_len,d_r]
        # 获得位置
        _,seq_len,_ = x.shape
        pos = torch.arange(0,seq_len,device=self.device).unsqueeze(1)# [seq_len,1]
        # 构建操作矩阵
        _t_2i = 2*torch.arange(0,self.d_r//2,device=self.device) # [d_r//2]
        _2i = torch.zeros(1,self.d_r,device=self.device) # [1,d_r]
        _2i[:,1::2] = _t_2i
        _2i[:,0::2] = _t_2i
        
        ro_cos = torch.cos(pos*torch.pow(self.ro_theta,(-_2i/self.d_r))) # [seq_len,d_r]
        ro_sin = torch.sin(pos*torch.pow(self.ro_theta,(-_2i/self.d_r)))

        # 处理x
        x_rohalf = torch.zeros_like(x,device=self.device) # [batch,seq_len,d_r]
        x_rohalf[:,:,1::2] = x[:,:,0::2]
        x_rohalf[:,:,0::2] = -x[:,:,1::2]

        x = x*ro_cos + x_rohalf*ro_sin
        return x

    def forward(self,q,kv,mask=None):
        # 降格
        c_q = self.w_dq(q.float())
        c_kv = self.w_dkv(kv.float())

        # 升格
        q_nope = self.w_uq(c_q)
        k_nope = self.w_uk(c_kv)
        v = self.w_uv(c_kv)

        # 额外的q、k头
        q_r = self.rope(self.w_qr(c_q)).unsqueeze(1).repeat(1,self.n_head,1,1)
        k_r = self.rope(self.w_kr(kv)).unsqueeze(1).repeat(1,self.n_head,1,1)

        # 拆分多头
        _,q_seq_len,_ = q_nope.shape
        _,k_seq_len,_ = v.shape
        d_h = self.d_model//self.n_head
        q_nope = q_nope.view(-1,q_seq_len,self.n_head,d_h).permute(0,2,1,3)
        k_nope = k_nope.view(-1,k_seq_len,self.n_head,d_h).permute(0,2,1,3)
        v = v.view(-1,k_seq_len,self.n_head,d_h).permute(0,2,1,3)

        # 完整的q、k
        q = torch.cat([q_nope,q_r],dim=-1)
        k = torch.cat([k_nope,k_r],dim=-1)
 
        score = (q@k.transpose(2,3))/math.sqrt(d_h+self.d_r)
        if mask is not None:
            score=score.masked_fill(mask==0,-1e4)
        attention = self.softmax(score)@v

        # 联合多头
        attention = attention.permute(0,2,1,3).contiguous().view(-1,q_seq_len,self.d_model)
        output = self.w_o(attention)
        return output

if __name__ == '__main__':
    MLA_layer = MLA (
        d_model=1024,
        d_c=512,
        d_r=16,
        n_head=16,
        device=torch.device('cpu'),
        ro_theta=10000,
    )
    all_time = 0
    for _ in range(30):
        mla_try_tensor = torch.randn(2,90,1024)
        start = time.perf_counter()
        output = MLA_layer(mla_try_tensor,mla_try_tensor)
        end = time.perf_counter()
        all_time += end-start
    print(f"平均用时:{((all_time)/30):.2f}s")
    print(output.shape)
    print(output)
