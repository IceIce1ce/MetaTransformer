import torch
from torch import nn
import math
from torch.nn import functional as F
from .video_swin_transformer import SwinTransformer3D

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Cls(nn.Module):
    def __init__(self, model, shape_input, shape_output, dim_latent, dropout=0.5, rnn_state_size=None, rnn_cell_num=1, rnn_type='lstm', avg_pool_shape=(1, 6, 6)):
        super(Cls, self).__init__()
        self.shape_input = shape_input
        self.shape_output = shape_output
        self.rnn_state_size = rnn_state_size
        self.rnn_type = rnn_type
        self.has_rnn_state = rnn_state_size is not None
        if self.has_rnn_state:
            rnn_cls = nn.LSTM
            self.rnn = rnn_cls(dim_latent, rnn_state_size, rnn_cell_num)
            self.rnn_bn = nn.LayerNorm(dim_latent)
        self.dim_state = self.shape_input[0] * math.prod(avg_pool_shape)
        self.lin1 = nn.Linear(self.dim_state, dim_latent)
        self.lin2 = nn.Linear(rnn_state_size or dim_latent, dim_latent)
        self.lin3 = nn.Linear(dim_latent, self.shape_output)
        self.downsize = nn.AdaptiveAvgPool3d(avg_pool_shape)
        self.bn = nn.LayerNorm(self.dim_state)
        self.drop = nn.Dropout(dropout)
        self.apply(weights_init_)
        self.model = model

    def forward(self, x, rnn_state=None):
        # Long-term memory module
        x = self.model(x) # [2, 1024, 2, 15, 20]
        x = self.downsize(x) # [2, 1024, 1, 6, 6]
        x = x.contiguous().view(x.shape[0], -1) # [2, 36864]
        x = self.bn(x) # [2, 36864]
        x = F.relu(self.lin1(x)) # [2, 1024]
        x = self.drop(x) # [2, 1024]
        # Short-term memory module
        if self.has_rnn_state:
            x = self.rnn_bn(x) # [2, 1024]
            x = x.unsqueeze(0) # [1, 2, 1024]
            x, rnn_state = self.rnn(x, rnn_state)
            x = x.squeeze(0)
            if self.rnn_type == 'lstm':
                hx, cx = rnn_state
                rnn_state = (hx.detach(), cx.detach())
            else:
                rnn_state = rnn_state.detach()
        x = F.relu(self.lin2(x)) # [2, 1024]
        x = self.drop(x)
        x = self.lin3(x) # [2, 2]
        return x, rnn_state

def build_cls(cfg, model, shape_input):
    shape_output = 2
    dim_latent = cfg.get('dim_latent', 1024)
    dropout = cfg.get('dropout', 0.5)
    rnn_state_size = cfg.get('rnn_state_size', None)
    rnn_cell_num = cfg.get('rnn_cell_num', 1)
    rnn_type = cfg.get('rnn_type', 'lstm')
    avg_pool_shape = (1, 6, 6)
    return Cls(model, shape_input, shape_output, dim_latent, dropout=dropout, rnn_state_size=rnn_state_size,
               rnn_cell_num=rnn_cell_num, rnn_type=rnn_type, avg_pool_shape=avg_pool_shape).to(cfg.device)

def build_model_cfg(cfg):
    # t_outshape should be incremented each time NF is over patch depth
    t_type = cfg.get('transformer_type', 'swin-t')
    t_outshape = ((cfg.NF-1) // 4) + 1 # output shape of transformer layer
    hw_outshape = [15, 20] # [H, W] -> output shape in spatial dimensions
    depths = cfg.get('depths', None)
    if t_type == 'swin-b':
        shape_input = [1024, t_outshape] + hw_outshape
        mod_kwargs = {'depths': depths or [2, 2, 18, 2], 'embed_dim': 128, 'num_heads': [4, 8, 16, 32], 'drop_path_rate': 0.3,
                      'patch_size': (2, 4, 4), 'window_size': (16, 7, 7), 'patch_norm': True, 'in_chans': 3}
    elif t_type == 'swin-t':
        shape_input = [768, t_outshape] + hw_outshape
        mod_kwargs = {'embed_dim': 96, 'depths': depths or [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24],
                      'drop_path_rate': 0.1, 'patch_size': (2, 4, 4), 'window_size': (8, 7, 7), 'mlp_ratio': 4, 'qkv_bias': True,
                      'qk_scale': None, 'drop_rate': 0., 'attn_drop_rate': 0., 'patch_norm': True, 'in_chans': 3}
    t_model = SwinTransformer3D
    return t_model, mod_kwargs, shape_input