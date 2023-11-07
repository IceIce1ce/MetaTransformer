import torch
from torch import nn
from torch.nn import functional as F
import itertools

class C3D(nn.Module):
    def __init__(self, pretrained=None):
        super(C3D, self).__init__()
        self.pretrained = pretrained
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        self.fc6 = nn.Linear(8192, 4096)
        self.relu = nn.ReLU()
        self.__init_weight()
        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        return x

    def __load_pretrained_weights(self):
        # ignore fc7 and gc8, just get fc6 for feature extraction
        corresp_name = {"features.0.weight": "conv1.weight", "features.0.bias": "conv1.bias", "features.3.weight": "conv2.weight", "features.3.bias": "conv2.bias",
                        "features.6.weight": "conv3a.weight", "features.6.bias": "conv3a.bias", "features.8.weight": "conv3b.weight", "features.8.bias": "conv3b.bias",
                        "features.11.weight": "conv4a.weight", "features.11.bias": "conv4a.bias", "features.13.weight": "conv4b.weight", "features.13.bias": "conv4b.bias",
                        "features.16.weight": "conv5a.weight", "features.16.bias": "conv5a.bias", "features.18.weight": "conv5b.weight", "features.18.bias": "conv5b.bias",
                        "classifier.0.weight": "fc6.weight", "classifier.0.bias": "fc6.bias"}
        ignored_weights = [f"{layer}.{type_}" for layer, type_ in itertools.product(['fc7', 'fc8'], ['bias', 'weight'])]
        p_dict = torch.load(self.pretrained)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name.values() and name in ignored_weights:
                continue
            s_dict[name] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

class c3d_lstm(nn.Module):
    def __init__(self, shape_output=2, dim_latent=4096, rnn_state_size=None, rnn_cell_num=1, rnn_type='lstm'):
        super(c3d_lstm, self).__init__()
        self.shape_output = shape_output
        self.rnn_type = rnn_type
        self.has_rnn_state = rnn_state_size is not None
        if self.has_rnn_state:
            self.rnn = nn.LSTM(dim_latent, rnn_state_size, rnn_cell_num)
            self.rnn_bn = nn.LayerNorm(dim_latent)
        self.lin1 = nn.Linear(rnn_state_size or dim_latent, dim_latent)
        self.lin2 = nn.Linear(dim_latent, self.shape_output)
        self.drop = nn.Dropout(0.5)
        self.apply(weights_init_)

    def forward(self, x, rnn_state=None): # [16, 3, 16, 112, 112] -> [B, C, T, H, W]
        # extract features from C3D module
        C3D_model = C3D(pretrained='pretrained/c3d.pickle').cuda()
        x = C3D_model(x) # [16, 4096]
        # Short-term memory module
        if self.has_rnn_state:
            x = self.rnn_bn(x) # [2, 1024]
            x = x.unsqueeze(0) # [1, 2, 1024]
            x, rnn_state = self.rnn(x, rnn_state)
            x = x.squeeze(0) # [16, 3, 16, 112, 112]
            if self.rnn_type == 'lstm':
                hx, cx = rnn_state
                rnn_state = (hx.detach(), cx.detach())
            else: # GRU
                rnn_state = rnn_state.detach()
        x = F.relu(self.lin1(x)) # [2, 1024]
        x = self.drop(x) # [2, 1024]
        x = self.lin2(x) # [2, 2]
        return x, rnn_state

def build_c3d_lstm():
    return c3d_lstm(shape_output=2, dim_latent=4096, rnn_state_size=1024, rnn_cell_num=1, rnn_type='lstm').cuda()