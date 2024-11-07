
import torch
import torch.nn as nn
import torchvision.models as models
from functools import reduce
from model.utils.main_blocks import conv_block, double_conv_block_a, double_conv_block, Upconv, params
from model.utils.dca import DCA

#ssh增量卷积进行信息聚合
class SSH(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSH, self).__init__()
        assert out_channels % 4 == 0
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv5x5_1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv5x5_2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv7x7_2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv7x7_3 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv3x3 = self.conv3x3(x)

        conv5x5_1 = self.conv5x5_1(x)
        conv5x5 = self.conv5x5_2(conv5x5_1)

        conv7x7_2 = self.conv7x7_2(conv5x5_1)
        conv7x7 = self.conv7x7_3(conv7x7_2)

        output = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        output = self.relu(output)

        return output


class FPN(nn.Module):
    def __init__(self,
                 num_class,
                 attention=True,
                 # attention=False,
                 n=1,
                 k=1,
                 input_size=(56, 56),
                 patch_size=8,
                 spatial_att=True,
                 channel_att=True,
                 spatial_head_dim=None,
                 channel_head_dim=None,
                 ):
        super(FPN, self).__init__()

        if channel_head_dim is None:
            channel_head_dim = [1, 1, 1, 1]
        if spatial_head_dim is None:
            spatial_head_dim = [4, 4, 4, 4]

        resnet = models.resnet34(pretrained=True)

        self.attention = attention
        patch = input_size[0] // patch_size

        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # Layer1
        self.layer2 = nn.Sequential(*list(resnet.children())[5])  # Layer2
        self.layer3 = nn.Sequential(*list(resnet.children())[6])  # Layer3
        self.layer4 = nn.Sequential(*list(resnet.children())[7])  # Layer4

        self.ssh1 = SSH(64, 256)
        self.ssh2 = SSH(128, 256)
        self.ssh3 = SSH(256, 256)
        self.ssh4 = SSH(512, 256)

        #  without ssh
        # self.conv1 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        if self.attention:
            self.DCA = DCA(n=n,
                                features = [int(64 * k), int(128 * k), int(256 * k), int(512 * k)],
                                strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                                patch=patch,
                                spatial_att=spatial_att,
                                channel_att=channel_att,
                                spatial_head=spatial_head_dim,
                                channel_head=channel_head_dim,)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.self_attention = MultiHeadSelfAttention(embed_size=256, heads=8)
        self.final_conv_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.final_batch_1 = nn.BatchNorm2d(512)
        self.final_relu_1 = nn.ReLU()
        self.final_conv_2 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.final_batch_2 = nn.BatchNorm2d(1024)
        self.final_relu_2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.maxpool = nn.MaxPool2d(2,2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # out_put=(1,1)
        # self.fc = nn.Linear(1024 * 14 * 14, 1024)# Assuming 7 classes for facial expressions
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_class)  # 更改模型，添加fc两层

    def forward(self, x):
        # Backbone outputs
        # print(x.shape)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        # print(f"c1:{c1.shape}")
        # print(f"c2:{c2.shape}")
        # print(f"c3:{c3.shape}")
        # print(f"c4:{c4.shape}")
        if self.attention:
            c1, c2, c3, c4 = self.DCA([c1, c2, c3, c4])
        # print(f"Ac1:{c1.shape}")
        # print(f"Ac2:{c2.shape}")
        # print(f"Ac3:{c3.shape}")
        # print(f"Ac4:{c4.shape}")

        # c1 - self.dropout(c1)
        # c2 - self.dropout(c2)
        # c3 - self.dropout(c3)
        # c4 - self.dropout(c4)
        # print(f"Ac1:{c1.shape}")
        # print(f"Ac2:{c2.shape}")
        # print(f"Ac3:{c3.shape}")
        # print(f"Ac4:{c4.shape}")
        # 使用ssh增量卷积进行信息聚合
        p4 = self.ssh4(c4)
        p3 = self.ssh3(c3) + self.upsample(p4)
        p2 = self.ssh2(c2) + self.upsample(p3)
        p1 = self.ssh1(c1) + self.upsample(p2)
        # # Feature Pyramid Network without ssh
        # p4 = self.conv4(c4)
        # p3 = self.conv3(c3) + self.upsample(p4)
        # p2 = self.conv2(c2) + self.upsample(p3)
        # p1 = self.conv1(c1) + self.upsample(p2)
        # print(f"p1:{p1.shape}")
        # print(f"p2:{p2.shape}")
        # print(f"p3:{p3.shape}")
        # print(f"p4:{p4.shape}")
        # print(f"up-p4:{self.upsample(p4).shape}")

        # #使用多头自注意力机制
        # p1 = p1.permute(0, 2, 3, 1).contiguous()  # Change shape to (batch_size, height, width, channels)
        # p1 = p1.view(p1.shape[0], -1, 256)  # Flatten spatial dimensions
        # p = self.self_attention(p1, p1, p1, mask=None)
        # print(f"p:{p.shape}")
        # p = p.view(p.shape[0], 8, 8, 256).permute(0, 3, 1, 2)  # Change back to (batch_size, channels, height, width)
        # print(f"p1:{p1.shape}")
        p = self.final_conv_1(p1)
        p = self.final_batch_1(p)
        p = self.final_relu_1(p)
        # print(f"p2:{p.shape}")
        p = self.maxpool(p)
        # print(f"p:{p.shape}")
        p = self.final_conv_2(p)
        p = self.final_batch_2(p)
        p = self.final_relu_2(p)
        # print(f"p3:{p.shape}")
        p = self.pool(p)
        # print(f"p4:{p.shape}")
        p = p.view(p.size(0), -1)
        # print(f"p5:{p.shape}")
        p = self.fc2(p)
        # print(f"p6:{p.shape}")
        p = self.dropout(p)
        p = self.fc3(p)
        print(p.shape)
        p = self.dropout(p)
        out = self.fc4(p)
        print(out.shape)
        return out
# # 多头自注意力机制
# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, embed_size, heads):
#         super(MultiHeadSelfAttention, self).__init__()
#         self.embed_size = embed_size
#         self.heads = heads
#         self.head_dim = embed_size // heads
#
#         assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"
#
#         self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
#
#     def forward(self, values, keys, query, mask):
#         N = query.shape[0]
#         value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
#
#         # Split the embedding into self.heads different pieces
#         values = values.reshape(N, value_len, self.heads, self.head_dim)
#         keys = keys.reshape(N, key_len, self.heads, self.head_dim)
#         queries = query.reshape(N, query_len, self.heads, self.head_dim)
#
#         values = self.values(values)
#         keys = self.keys(keys)
#         queries = self.queries(queries)
#
#         energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, float("-1e20"))
#
#         attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
#
#         out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
#             N, query_len, self.heads * self.head_dim
#         )
#
#         out = self.fc_out(out)
#         return out


# class FPN_sk(nn.Module):
#     def __init__(self):
#         super(FPN_sk, self).__init__()
#
#         resnet = models.resnet50(weights="IMAGENET1K_V1")
#
#         self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # Layer1
#         self.layer2 = nn.Sequential(*list(resnet.children())[5])  # Layer2
#         self.layer3 = nn.Sequential(*list(resnet.children())[6])  # Layer3
#         self.layer4 = nn.Sequential(*list(resnet.children())[7])  # Layer4
#         self.conv1 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
#
#         self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
#         self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
#                                  nn.BatchNorm2d(d),
#                                  nn.ReLU(inplace=True))  # 降维
#         self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
#         self.softmax = nn.Softmax(dim=1)
#
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         # self.self_attention = MultiHeadSelfAttention(embed_size=256, heads=8)
#         # self.final_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.fc = nn.Linear(256 * 8 * 8, 7)  # Assuming 7 classes for facial expressions
#
#     def forward(self, x):
#         # Backbone outputs
#         c1 = self.layer1(x)
#         c2 = self.layer2(c1)
#         c3 = self.layer3(c2)
#         c4 = self.layer4(c3)
#
#
#         # Feature Pyramid Network
#         # p4 = self.conv4(c4)
#         # p3 = self.conv3(c3) + self.upsample(p4)
#         # p2 = self.conv2(c2) + self.upsample(p3)
#         # p1 = self.conv1(c1) + self.upsample(p2)
#         # print(f"p1:{p1.shape}")
#         #使用多头自注意力机制
#         # p1 = p1.permute(0, 2, 3, 1).contiguous()  # Change shape to (batch_size, height, width, channels)
#         # p1 = p1.view(p1.shape[0], -1, 256)  # Flatten spatial dimensions
#         # p = self.self_attention(p1, p1, p1, mask=None)
#         # p = p.view(p.shape[0], 8, 8, 256).permute(0, 3, 1, 2)  # Change back to (batch_size, channels, height, width)
#         # p = self.final_conv(p)
#         # print(f"p:{p.shape}")
#         # p = p1.view(p1.size(0), -1)
#         # print(p.shape)
#         # output = self.fc(p)
#         # return output
#         U = c1 + c2 + c3 + c4?
#         s = self.global_pool(U)
#         z = self.fc1(s)
#         a_b = self.fc2(z)
#         a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
#         a_b = self.softmax(a_b)
#         a_b = list(a_b.chunk(self.M, dim=1))
#         a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1),
#                        a_b))
#         V = list(map(lambda x, y: x * y, output,
#                      a_b))
#         V = reduce(lambda x, y: x + y,
#                    V)
#         return V



if __name__ == '__main__':
    from train import RecorderMeter
    inputs = torch.rand([1,3, 224, 224])
    #b, c, h, w = inputs.shape
    # model_ft = FPN(num_class=7)
    # # model_ft.load_state_dict(torch.load('D://桌面//FPN//[10-23]-[10-02]-model_best.pth', map_location='cpu'), strict=False)
    # model_features = nn.Sequential(*list(model_ft.modules())[:-4])
    # # print(model_features)
    # ou = model_features(inputs)
    # print(ou.size())
    model = FPN(num_class=7)
    # # model_features = model
    # # model_features.fc3 = nn.Identity()  # 去掉全局平均池化层
    # # model_features.fc4 = nn.Identity()
    # # model_features.fc2 = nn.Identity()
    # # model_features.dropout = nn.Identity()
    # # model_features.maxpool = nn.Identity()
    # # model_features.pool = nn.Identity()
    # # print(model_features)
    # # # inputs = torch.rand([8, 128, 28, 28])
    # # # model = SKNet(128,512)
    out = model(inputs)
    # print(out.size())
    # print(model)
    # 去掉最后两层
    # model.fc3 = nn.Identity()  # 去掉全局平均池化层
    # model.fc4 = nn.Identity()  # 去掉全连接层
    # 检查模型结构，确认已删除
    # print(model)
    # from PIL import Image
    # from torchvision import transforms
    # img_path = 'D:\\桌面\\FPN\\24026.jpg'  # 单张测试
    # img = Image.open(img_path).convert('RGB')
    # data_transform = {
    #     "val": transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])
    # }
    # img_tensor = data_transform['val'](img).unsqueeze(0)  # [1,3,224,224]
    # model = FPN(num_class=7)
    # out = model(img_tensor)
    # print(out.size())
    # print(model)