from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np
from vqvae import VQEmbedding
from vqloss import *
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')


# DSH(CVPR2016)
# paper [Deep Supervised Hashing for Fast Image Retrieval](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf)
# code [DSH-pytorch](https://github.com/weixu000/DSH-pytorch)
# code [CV_Project](https://github.com/aarathimuppalla/CV_Project)
# code [DSH_tensorflow](https://github.com/yg33717/DSH_tensorflow)

def get_config():
    config = {
        "alpha": 0.1,
        "optimizer":{"type":  optim.Adam, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        # "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[DSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": VQ_AlexNet_slip,
        # "net":ResNet,
        # "dataset": "cifar10",
        "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "voc2012",
        # "dataset": "imagenet",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 1000,
        "test_map": 15,
        "save_path": "save/vqDSH_slip",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [256],
    }
    config = config_dataset(config)
    return config


class DSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DSHLoss, self).__init__()
        self.m = 2 * bit
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        dist = (u.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(dim=2)
        y = (y @ self.Y.t() == 0).float()

        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()
        loss2 = config["alpha"] * (1 - u.abs()).abs().mean()

        return loss1 + loss2

class vqdshLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(vqdshLoss, self).__init__()
        self.center_criterion= CenterLoss(num_classes=10, feat_dim=256, use_gpu=True)
        self.fc = torch.nn.Linear(256, 10, bias=False).to(config["device"])
        self.criterion = torch.nn.CrossEntropyLoss().to(config["device"])
        self.trplet = TripletLoss(0.3)
    def forward(self,feat_hard,feat_soft,feat,target):
        label = target.argmax(axis=1)
        pre_label = self.fc(feat_hard)
        # print(label.shape)
        # print(pre_label.shape)
        loss1 = self.criterion(pre_label,label)
        # print(cls_sorc_hard.shape)
        # print(target.shape)
        # print("!!!!!")
        loss4 = F.mse_loss(feat_hard,feat_soft)
        return loss1+loss4
def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    # optimizer = optim.Adam(lr=0.0002,weight_decay=10 ** -5)
    criterion = vqdshLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            q_feat_soft,q_feat_hard,feat= net(image)
            # print(cls_hard.shape)
            # print(label.shape)
            # print("!!!!!")
            loss = criterion(q_feat_hard,q_feat_soft, feat,label.float())
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_vqdsh(test_loader, net, device=device)

            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_vqdsh(dataset_loader, net, device=device)

            # print("calculating map.......")
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])

            if mAP > Best_mAP:
                Best_mAP = mAP
                if "cifar10-1" == config["dataset"] and epoch > 29:
                    P, R = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy())
                    print(f'Precision Recall Curve data:\n"DSH":[{P},{R}],')
                if "save_path" in config:
                    if not os.path.exists(config["save_path"]):
                        os.makedirs(config["save_path"])
                    print("save in ", config["save_path"])
                    np.save(os.path.join(config["save_path"], config["dataset"] + str(mAP) + "-" + "trn_binary.npy"),
                            trn_binary.numpy())
                    torch.save(net.state_dict(),
                               os.path.join(config["save_path"], config["dataset"] + "-" + str(mAP) + "-model.pt"))
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP))
            print(config)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
