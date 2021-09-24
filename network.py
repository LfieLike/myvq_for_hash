import torch.nn as nn
from torchvision import models
import torch.nn.init as init
from vqvae import VQEmbedding
import torch
class AlexNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        # print(x.shape)
        return x


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet50"):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[res_model](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y

class DSH(nn.Module):
    def __init__(self, num_binary=120):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # same padding
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, num_binary)
        )

        for m in self.modules():
            if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class VQ_AlexNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(VQ_AlexNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias
        self.vq = VQEmbedding(256,128)
        self.cls = nn.Linear(256,10)
        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        q_feat_soft,q_feat_hard = self.vq.straight_through(x)
        # cls_soft = self.cls(q_feat_soft)
        # cls_hard = self.cls(q_feat_hard)
        return q_feat_soft,q_feat_hard,x

class VQ_AlexNet_slip(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(VQ_AlexNet_slip, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias
        self.vq1 = VQEmbedding(64,128)
        self.vq2 = VQEmbedding(64,128)
        self.cls = nn.Linear(256,10)
        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        x1 = x[:,:128]
        x2 = x[:,128:]
        q_feat_soft1,q_feat_hard1 = self.vq1.straight_through(x1)
        q_feat_soft2,q_feat_hard2 = self.vq2.straight_through(x2)
        # cls_soft = self.cls(q_feat_soft)
        # cls_hard = self.cls(q_feat_hard)

        q_feat_soft = torch.cat((q_feat_soft1,q_feat_soft2),1)
        q_feat_hard = torch.cat((q_feat_hard1,q_feat_hard2),1)
        # print(q_feat_soft.shape)
        return q_feat_soft,q_feat_hard,x
