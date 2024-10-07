import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return base

if __name__ == '__main__':
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    # ---------------------------------------------------------------------- #
    loadnet = torch.load('./weights/SRVGGNetCompact.pth', map_location=torch.device('cpu'))
    model.load_state_dict(loadnet, True)
    # ------------------------ 读取官方pth并导出 ----------------------------- #
    # model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    # model_path = "/home/scz/CODE/水膜/Real-ESRGAN-master/weights/realesr-general-wdn-x4v3.pth"
    # loadnet = torch.load(model_path, map_location=torch.device('cpu'))
    # model.load_state_dict(loadnet['params'],True)
    #
    # def extract_weights(model):
    #     all_weights = []
    #
    #     for name, param in model.named_parameters():
    #         if 'weight' in name:
    #             if len(param.shape) == 4:
    #                 weight = param.permute(2,3,1,0).detach().cpu().numpy()
    #             else:
    #                 weight = param.detach().cpu().numpy()
    #             all_weights.append(weight.flatten())
    #         elif 'bias' in name:
    #             bias = param.detach().cpu().numpy()
    #             all_weights.append(bias.flatten())
    #     # 64 * 28
    #     combined_weights = np.concatenate(all_weights)
    #     return combined_weights
    #
    # # 提取权重并保存到二进制文件
    # weights = extract_weights(model)
    # weights.tofile('real_esrgan_param.bin')
    # ---------------------------------------------------------------------- #
    img = cv2.imread('Your img path', cv2.IMREAD_ANYCOLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img).float()/255
    if img is None:
        print("Error: Unable to open image file!")
    else:
        # 输出图像的深度和通道数
        print("Image depth: ", img.dtype)
        print("Number of channels: ", img.shape[2] if len(img.shape) == 3 else 1)
    x = x.permute(2,0,1).unsqueeze(0)
    # ---------------------------------------------------------------------- #
    start_time = time.time()  # 记录开始时间

    with torch.no_grad():  # 禁用梯度
        y = model(x)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算时间差

    print(f"运行时间: {elapsed_time:.6f} 秒")
    summary = torch.cuda.memory_summary(device=0, abbreviated=False)
    print(summary)
    # ------------------------------------------------------------- #
    y = y.squeeze(0).data.squeeze().float().cpu().clamp_(0, 1)  # 去除批次维度，移动到CPU，并将像素值限制在[0, 1]范围内
    y = y.permute(1, 2, 0).numpy()  # 将通道维度移动到最后，并转换为NumPy数组
    # 将图像数据从[0, 1]范围恢复到[0, 255]范围，并转换为uint8类型
    y = (y * 255).astype(np.uint8)

    # 保存图像
    cv2.imwrite('Save path', cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
