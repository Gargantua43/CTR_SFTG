import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from Dataset_pair_train_id import MyDatasetTrain
from torch.utils.data import DataLoader


# 参数初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class RankNet(torch.nn.Module):
    def __init__(self):
        super(RankNet, self).__init__()
        # 视频网络
        self.fc_visual = torch.nn.Linear(1280, 128)
        # 音频网络
        self.fc_audio = torch.nn.Linear(128, 128)
        # Attention网络
        self.fc_attention_visual = torch.nn.Linear(128, 1)
        self.fc_attention_audio = torch.nn.Linear(128, 1)
        self.attention_sigmoid = torch.nn.Sigmoid()
        # 融合网络
        self.fc_fusion = torch.nn.Linear(257, 1)
        # 添加Dropout层
        self.dropout = torch.nn.Dropout(p=0.5)
        # 预测概率公式Sigmoid
        self.calculate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x_visual1, x_audio1, x_visual2, x_audio2, pair_id):
        # input_visual_embedding1
        y_visual_fc1 = F.relu(self.fc_visual(x_visual1))
        y_visual_drop1 = self.dropout(y_visual_fc1)
        # attention_visual_embedding1
        y_visual_attention1 = self.attention_sigmoid(self.fc_attention_visual(y_visual_drop1))
        y_visual_weight1 = F.softmax(y_visual_attention1, dim=1)
        y_visual_weight_emb1 = torch.bmm(y_visual_drop1.transpose(1, 2), y_visual_weight1).squeeze(2)

        # input_audio_embedding1
        y_audio_fc1 = F.relu(self.fc_audio(x_audio1))
        y_audio_drop1 = self.dropout(y_audio_fc1)
        # attention_audio_embedding1
        y_audio_attention1 = self.attention_sigmoid(self.fc_attention_audio(y_audio_drop1))
        y_audio_weight1 = F.softmax(y_audio_attention1, dim=1)
        y_audio_weight_emb1 = torch.bmm(y_audio_drop1.transpose(1, 2), y_audio_weight1).squeeze(2)

        # id增加一个维度保证后续的计算正常运行
        pair_id1 = torch.unsqueeze(pair_id, 1)

        # connect_visual&audio and predict_CTR
        connect_feature1 = torch.cat([y_visual_weight_emb1, y_audio_weight_emb1], dim=1)
        connect_id1 = torch.cat([connect_feature1, pair_id1], 1)
        y_ctr1 = self.fc_fusion(connect_id1)

        # input_visual_embedding2
        y_visual_fc2 = F.relu(self.fc_visual(x_visual2))
        y_visual_drop2 = self.dropout(y_visual_fc2)
        # attention_visual_embedding2
        y_visual_attention2 = self.attention_sigmoid(self.fc_attention_visual(y_visual_drop2))
        y_visual_weight2 = F.softmax(y_visual_attention2, dim=1)
        y_visual_weight_emb2 = torch.bmm(y_visual_drop2.transpose(1, 2), y_visual_weight2).squeeze(2)

        # input_audio_embedding2
        y_audio_fc2 = F.relu(self.fc_audio(x_audio2))
        y_audio_drop2 = self.dropout(y_audio_fc2)
        # attention_audio_embedding2
        y_audio_attention2 = self.attention_sigmoid(self.fc_attention_audio(y_audio_drop2))
        y_audio_weight2 = F.softmax(y_audio_attention2, dim=1)
        y_audio_weight_emb2 = torch.bmm(y_audio_drop2.transpose(1, 2), y_audio_weight2).squeeze(2)

        # id增加一个维度保证后续的计算正常运行
        pair_id2 = torch.unsqueeze(pair_id, 1)

        # connect_visual&audio and predict_CTR
        connect_feature2 = torch.cat([y_visual_weight_emb2, y_audio_weight_emb2], dim=1)
        connect_id2 = torch.cat([connect_feature2, pair_id2], 1)
        y_ctr2 = self.fc_fusion(connect_id2)

        # output_calculate_value
        dif = y_ctr1 - y_ctr2
        calculate_ctr = self.calculate_sigmoid(dif)

        return calculate_ctr, y_ctr1, y_ctr2


# # x_visual_embedding1 = torch.randn(2, 116, 1280)
# # x_visual_embedding2 = torch.randn(2, 116, 1280)
# # x_audio_embedding1 = torch.randn(2, 39, 128)
# # x_audio_embedding2 = torch.randn(2, 39, 128)
# #
# # model = RankNet()
# # # y_visual_embedding = model(x_visual_embedding1, x_audio_embedding1)
# # output = model(x_visual_embedding1, x_audio_embedding1, x_visual_embedding2, x_audio_embedding2)
# #
# # # print(output)
# # print('--------')
# # print(output[0])
# # print(output[1])
# # print(output[2])
#
#
# # print(output.shape)
#
# def seed_torch(seed=1):
#     random.seed(seed)  # Python random module
#     # os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
#     np.random.seed(seed)  # Numpy module
#     torch.manual_seed(seed)  # 为CPU设置随机数种子
#     torch.cuda.manual_seed(seed)  # 为当前GPU设置随机数种子
#     torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机数种子
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#
#
# seed_torch()
#
# my_dataset = MyDatasetTrain("./SFTG_CTR_label_pair_train_id.csv")
# train_loader = DataLoader(dataset=my_dataset, batch_size=2)
#
# model_rank = RankNet()
# model_rank = model_rank.double()
# model_rank = model_rank.train()
# model_rank.apply(init_weights)
#
# for v1_train, a1_train, v2_train, a2_train, ctr_label1, id_pair in iter(train_loader):
#     output = model_rank(v1_train, a1_train, v2_train, a2_train, id_pair)
#
#     print(output[0])
#     print(output[1])
#     print(output[2])
#     print('++++++++++')
