"""
双卡测试
代码描述：
    Fix_attention+SFTG平台数据(不带有平台ID)+Adam=0.01+num_works=6+all_batch_size=1024
    使用修改过的Snap_visual_embedding(一张一张按照图片标号提取特征)
    双卡跑出来的权重文件：epoch= val_acc=

注：
    此程序若有参数修改，以双卡机器为准
"""
import torch
import numpy as np
import random
import os
import torch.optim as optim
import pandas as pd

# from Fusion_Net_revise import RankNet
from Fusion_Net_revise_attention_id import RankNet, init_weights
from Dataset_pair_test_id import MyDatasetTest
from torch.utils.data import DataLoader


def seed_torch(seed=1):
    random.seed(seed)  # Python random module
    # os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # 为CPU设置随机数种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机数种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机数种子
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


seed_torch()


def main():
    # Available_GPU_List
    device_ids = [0, 1]
    one_gpu_batch = 512

    # 导入测试集
    my_dataset_test = MyDatasetTest("./SFTG_CTR_label_pair_test_id.csv")
    test_loader = DataLoader(dataset=my_dataset_test, batch_size=one_gpu_batch * len(device_ids), drop_last=True,
                             num_workers=8)
    test_num = len(my_dataset_test)
    print("测试集数量:{}".format(test_num))

    # 载入模型权重
    model_weight_path = './'

    # 实例化模型
    model_rank = RankNet()
    model_rank = model_rank.double()
    model_rank = model_rank.train()
    model_rank = torch.nn.DataParallel(model_rank, device_ids=device_ids)

    # 将模型丢到cuda训练
    if torch.cuda.is_available():
        print("cuda是否能够使用:", torch.cuda.is_available())
        model_rank = model_rank.cuda(device=device_ids[0])

    # test
    pre_right = 0
    acc = 0.0
    test_steps = 0
    model_rank.load_state_dict(torch.load(model_weight_path))
    model_rank = model_rank.eval()
    with torch.no_grad():
        for v1_test, a1_test, v2_test, a2_test, ctr_label_test in iter(test_loader):
            # 用来记录第几个batch
            test_steps += 1
            print("第{}个batch".format(test_steps))

            # 将验证数据丢到cuda上训练
            if torch.cuda.is_available():
                v1_test = v1_test.cuda(device=device_ids[0])
                a1_test = a1_test.cuda(device=device_ids[0])
                v2_test = v2_test.cuda(device=device_ids[0])
                a2_test = a2_test.cuda(device=device_ids[0])
                ctr_label_test = ctr_label_test.cuda(device=device_ids[0])

            # 网络输出
            output_test = model_rank(v1_test, a1_test, v2_test, a2_test)

            # 计算acc
            for i in range(1024):
                if output_test[1][i] > output_test[2][i]:
                    p = 1
                elif output_test[1][i] < output_test[2][i]:
                    p = 0
                else:
                    p = 0.5

                # 与Label进行比较,并且统计预测正确的个数
                if p == ctr_label_test[i]:
                    pre_right += 1
            print("第{}个batch:COMPARE_FINISH!".format(test_steps))
            print("当前预测正确的个数是:{}".format(pre_right))
        # 计算准确率
        acc = pre_right / test_num
        print("SFTG+With_ID+RankNet准确率是:{}".format(acc))


if __name__ == '__main__':
    main()
