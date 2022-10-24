import torch
import torch.nn as nn
from timm.models.layers import to_2tuple,DropPath

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels,3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    ) 

def conv_bank(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels,3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    ) 

def window_partition(x, window_size):
    B, H, W, C = x.shape
    
    num_windows= (H//window_size)*(W//window_size)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B,num_windows,window_size, window_size, C)

    x1=x[:,num_windows-(num_windows),:,:]
    x1 = x1.permute(0, 3,1,2)
    x2=x[:,num_windows-(num_windows-1),:,:]
    x2 = x2.permute(0, 3,1,2)
    x3=x[:,num_windows-(num_windows-2),:,:]
    x3 = x3.permute(0, 3,1,2)
    x4=x[:,num_windows-(num_windows-3),:,:]
    x4 = x4.permute(0, 3,1,2)
    
    return x1,x2,x3,x4

class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  
        qkv = self.qkv(x)
        qkv=qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        qkv=qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class MY_1(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.window_partition=window_partition
                
        self.conv0_1 = double_conv(3,32)
        self.conv0_2 = double_conv(3,32)
        self.conv0_3 = double_conv(3,32)
        self.conv0_4 = double_conv(3,32)  
        
        self.conv1_1 = double_conv(32,64)
        self.conv1_2 = double_conv(32,64)
        self.conv1_3 = double_conv(32,64)
        self.conv1_4 = double_conv(32,64)
        
        self.conv2_1 = double_conv(64,128)
        self.conv2_2 = double_conv(64,128)
        self.conv2_3 = double_conv(64,128)
        self.conv2_4 = double_conv(64,128)
        
        self.conv3_1 = double_conv(128,256)
        self.conv3_2 = double_conv(128,256)
        self.conv3_3 = double_conv(128,256)
        self.conv3_4 = double_conv(128,256)
          
        self.conv4_1 = double_conv(256,512)
        self.conv4_2 = double_conv(256,512)
        self.conv4_3 = double_conv(256,512)
        self.conv4_4 = double_conv(256,512)
        
        self.activation = torch.nn.Sigmoid()
        
        self.conv_B0 = conv_bank(512,256)
        self.conv_B1 = conv_bank(512,256)
        self.conv_B2 = conv_bank(384,128)
        self.conv_B3 = conv_bank(192,64)
        self.conv_B4 = conv_bank(96,32)
        
        self.conv_ =nn.Conv2d(32,1,1)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        
        
        dpr = [x.item() for x in torch.linspace(0,0,2)]  # stochastic depth decay rule
        
        self.blocks1 = nn.ModuleList([
            Block(
                dim=196, num_heads=7, mlp_ratio=1, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(2)])
        
        self.blocks2 = nn.ModuleList([
            Block(
                dim=196, num_heads=7, mlp_ratio=1, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(2)])
        
        self.blocks3 = nn.ModuleList([
            Block(
                dim=784, num_heads=7, mlp_ratio=1, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(2)])
                
        
    def forward(self, x_in):
        x1,x2,x3,x4=self.window_partition(x_in,112)
            
        conv01=self.conv0_1(x1)
        conv02=self.conv0_2(x2)
        conv03=self.conv0_3(x3)
        conv04=self.conv0_4(x4)
        
        conv01= self.maxpool(conv01)
        conv02= self.maxpool(conv02)
        conv03= self.maxpool(conv03)
        conv04= self.maxpool(conv04)
    
        conv11=self.conv1_1(conv01)
        conv12=self.conv1_2(conv02)
        conv13=self.conv1_3(conv03)
        conv14=self.conv1_4(conv04)
        
        conv11= self.maxpool(conv11)
        conv12= self.maxpool(conv12)
        conv13= self.maxpool(conv13)
        conv14= self.maxpool(conv14)
        
        
        conv21=self.conv2_1(conv11)
        conv22=self.conv2_2(conv12)
        conv23=self.conv2_3(conv13)
        conv24=self.conv2_4(conv14)
        
        conv21= self.maxpool(conv21)
        conv22= self.maxpool(conv22)
        conv23= self.maxpool(conv23)
        conv24= self.maxpool(conv24)
        
        
        
        ## for attention ###
        Before=conv21
        x=torch.cat([conv21,conv22,conv23,conv24],axis=2)
        shape_1=x.shape
        x=x.flatten(2)
        
        ## do attention here ####
        
        for blk in self.blocks3:
            x = blk(x)        
        x = x.view(shape_1)
        
        uncatted = torch.chunk(x,chunks=4,dim=2)
        After=uncatted[0]
        
        conv21=conv21+uncatted[0]
        conv22=conv22+uncatted[1]
        conv23=conv23+uncatted[2]
        conv24=conv24+uncatted[3]
        
        conv31=self.conv3_1(conv21)
        conv32=self.conv3_2(conv22)
        conv33=self.conv3_3(conv23)
        conv34=self.conv3_4(conv24)
        
        
        
        conv31= self.maxpool(conv31)
        conv32= self.maxpool(conv32)
        conv33= self.maxpool(conv33)
        conv34= self.maxpool(conv34)
        
        ## for attention ###

        x=torch.cat([conv31,conv32,conv33,conv34],axis=2)
        shape_1=x.shape
        x=x.flatten(2)
        
        ## do attention here ####
        
        for blk in self.blocks2:
            x = blk(x)        
        x = x.view(shape_1)
        
        uncatted = torch.chunk(x,chunks=4,dim=2)
        
        conv31=conv31+uncatted[0]
        conv32=conv32+uncatted[1]
        conv33=conv33+uncatted[2]
        conv34=conv34+uncatted[3]
        
        conv41=self.conv4_1(conv31)
        conv42=self.conv4_2(conv32)
        conv43=self.conv4_3(conv33)
        conv44=self.conv4_4(conv34)
        
        ## for attention ###
        
        
        x=torch.cat([conv41,conv42,conv43,conv44],axis=2)
        shape_1=x.shape
        x=x.flatten(2)
        
        ## do attention here ####
        for blk in self.blocks1:
            x = blk(x)        
        
        x = x.view(shape_1)
        uncatted = torch.chunk(x,chunks=4,dim=2)
        
        
        c_1=torch.cat([uncatted[0],uncatted[1]],axis=3)
        c_2=torch.cat([uncatted[2],uncatted[3]],axis=3)
        c=torch.cat([c_1,c_2],axis=2)
                
            ## decoder ###

        c = self.upsample(c)
        y=self.conv_B0(c)
        
        c_1=torch.cat([conv31,conv32],axis=3)
        c_2=torch.cat([conv33,conv34],axis=3)
        c=torch.cat([c_1,c_2],axis=2) 
        c = self.upsample(c)
        c=torch.cat([c,y],axis=1)
        y=self.conv_B1(c)


        c_1=torch.cat([conv21,conv22],axis=3)
        c_2=torch.cat([conv23,conv24],axis=3)
        c=torch.cat([c_1,c_2],axis=2)
        c=torch.cat([c,y],axis=1)
        c = self.upsample(c)
        
        y=self.conv_B2(c)
            
        c_1=torch.cat([conv11,conv12],axis=3)
        c_2=torch.cat([conv13,conv14],axis=3)
        c=torch.cat([c_1,c_2],axis=2)
        c=torch.cat([c,y],axis=1)
        
        c = self.upsample(c)
        y=self.conv_B3(c)
        
            
        c_1=torch.cat([conv01,conv02],axis=3)
        c_2=torch.cat([conv03,conv04],axis=3)
        c=torch.cat([c_1,c_2],axis=2)
        c=torch.cat([c,y],axis=1)
        c = self.upsample(c)
        
        y=self.conv_B4(c)
        out1 = self.conv_(y)    
        out1=self.activation(out1)
        return out1 ,Before,After
     
# def model() -> MY_1:
#     model = MY_1()
#     return model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, (224,224,3))
        