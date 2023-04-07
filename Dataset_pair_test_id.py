import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader


class MyDatasetTest(Dataset):
    def __init__(self, pair_csv_path):
        self.Pair_csv_path = pair_csv_path
        self.df_Pair = pd.read_csv(self.Pair_csv_path, encoding='utf-8')

        # self.Audio_csv_path = audio_csv_path
        # self.Visual_csv_path = visual_csv_path
        # self.df_Audio = pd.read_csv(self.Audio_csv_path, encoding='utf-8')
        # self.df_Visual = pd.read_csv(self.Visual_csv_path, encoding='utf-8')

    # data = np.loadtxt(csv_path, delimiter=',')
    # self.len = data.shape[0]
    # self.train_data = torch.from_numpy(data[:, :])
    # print("数据以准备好.....")
    # print(self.train_data)

    def __getitem__(self, index):
        # 读pair文件中Visual1一列
        visual_csv_dir1 = self.df_Pair['AD_CTR_visual_pair1_test'][index]
        visual_data1 = np.loadtxt(visual_csv_dir1, delimiter=',')
        visual_train1 = torch.from_numpy(visual_data1[:, :])

        # 读pair文件中Audio1一列
        audio_csv_dir1 = self.df_Pair['AD_CTR_audio_pair1_test'][index]
        audio_data1 = np.loadtxt(audio_csv_dir1, delimiter=',')
        audio_train1 = torch.from_numpy(audio_data1[:, :])

        # 读pair文件中Visual2一列
        visual_csv_dir2 = self.df_Pair['AD_CTR_visual_pair2_test'][index]
        visual_data2 = np.loadtxt(visual_csv_dir2, delimiter=',')
        visual_train2 = torch.from_numpy(visual_data2[:, :])

        # 读pair文件中Audio2一列
        audio_csv_dir2 = self.df_Pair['AD_CTR_audio_pair2_test'][index]
        audio_data2 = np.loadtxt(audio_csv_dir2, delimiter=',')
        audio_train2 = torch.from_numpy(audio_data2[:, :])

        # 读pair文件中label一列
        pair_label = self.df_Pair['CTR_Label'][index]
        label_data = np.array(pair_label)
        label_train = torch.from_numpy(label_data)

        # 读pair文件中id一列
        pair_id = self.df_Pair['SFTG_Test_id'][index]
        id_data = np.array(pair_id)
        id_train = torch.from_numpy(id_data)

        return visual_train1, audio_train1, visual_train2, audio_train2, label_train, id_train
        # return self.train_data[index]

    def __len__(self):
        # audio_len = len(self.df_Audio)
        # visual_len = len(self.df_Visual)
        pair_len = len(self.df_Pair)
        return pair_len


# my_dataset = MyDatasetTest("./SFTG_CTR_label_pair_test_id.csv")
# train_loader = DataLoader(dataset=my_dataset, batch_size=1)
# print(len(my_dataset))
# # train_loader = tqdm(train_loader)
# i = 0
# for v1_train, a1_train, v2_train, a2_train, label, id_train in iter(train_loader):
#     print("第{}个广告对".format(i))
#     print(v1_train.shape)
#     print(a1_train.shape)
#     print(v2_train.shape)
#     print(a2_train.shape)
#     print(label.shape)
#     print(id_train)
#     i += 1
#     # pass
#     # print('++++++++++++')
