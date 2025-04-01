import torch.nn as nn
import pandas as pd


class biomaker(nn.Module):
    def __init__(self, feature_names_lists, std_devs_list):
        super(biomaker, self).__init__()
        self.feature_names_autoencoder1 = feature_names_lists[0]
        self.feature_names_autoencoder2 = feature_names_lists[1]
        self.feature_names_autoencoder3 = feature_names_lists[2]

        self.std_devs_autoencoder1 = std_devs_list[0].set_index('Feature')['Standard Deviation']
        self.std_devs_autoencoder2 = std_devs_list[1].set_index('Feature')['Standard Deviation']
        self.std_devs_autoencoder3 = std_devs_list[2].set_index('Feature')['Standard Deviation']

    def forward(self, model):

        W1 = model.autoencoder1.get_first_layer_weights().detach().cpu().numpy()
        W2 = model.autoencoder2.get_first_layer_weights().detach().cpu().numpy()
        W3 = model.autoencoder3.get_first_layer_weights().detach().cpu().numpy()

        c1 = W1.sum(axis=0)
        c2 = W2.sum(axis=0)
        c3 = W3.sum(axis=0)

        weighted_sums1 = pd.Series(c1, index=self.feature_names_autoencoder1).multiply(self.std_devs_autoencoder1,
                                                                                       fill_value=0)
        weighted_sums2 = pd.Series(c2, index=self.feature_names_autoencoder2).multiply(self.std_devs_autoencoder2,
                                                                                       fill_value=0)
        weighted_sums3 = pd.Series(c3, index=self.feature_names_autoencoder3).multiply(self.std_devs_autoencoder3,
                                                                                       fill_value=0)
        df1 = pd.DataFrame({'Feature': weighted_sums1.index, 'Weighted_Sum': weighted_sums1.values})
        df2 = pd.DataFrame({'Feature': weighted_sums2.index, 'Weighted_Sum': weighted_sums2.values})
        df3 = pd.DataFrame({'Feature': weighted_sums3.index, 'Weighted_Sum': weighted_sums3.values})

        sorted_df1 = df1.sort_values(by='Weighted_Sum', ascending=False).head(30)
        sorted_df2 = df2.sort_values(by='Weighted_Sum', ascending=False).head(30)
        sorted_df3 = df3.sort_values(by='Weighted_Sum', ascending=False).head(30)

        return sorted_df1, sorted_df2, sorted_df3

class biomakera(nn.Module):
    def __init__(self, feature_names_lists, std_devs_list):
        super(biomakera, self).__init__()
        self.feature_names_autoencoder1 = feature_names_lists[0]
        self.feature_names_autoencoder2 = feature_names_lists[1]
        self.feature_names_autoencoder3 = feature_names_lists[2]

        self.std_devs_autoencoder1 = std_devs_list[0].set_index('Feature')['Standard Deviation']
        self.std_devs_autoencoder2 = std_devs_list[1].set_index('Feature')['Standard Deviation']
        self.std_devs_autoencoder3 = std_devs_list[2].set_index('Feature')['Standard Deviation']

    def forward(self, model):

        W1 = model.autoencoder1.get_first_layer_weights().detach().cpu().numpy()
        W2 = model.autoencoder2.get_first_layer_weights().detach().cpu().numpy()
        W3 = model.autoencoder3.get_first_layer_weights().detach().cpu().numpy()

        c1 = W1.sum(axis=0)
        c2 = W2.sum(axis=0)
        c3 = W3.sum(axis=0)

        weighted_sums1 = pd.Series(c1, index=self.feature_names_autoencoder1).multiply(self.std_devs_autoencoder1,
                                                                                       fill_value=0)
        weighted_sums2 = pd.Series(c2, index=self.feature_names_autoencoder2).multiply(self.std_devs_autoencoder2,
                                                                                       fill_value=0)
        weighted_sums3 = pd.Series(c3, index=self.feature_names_autoencoder3).multiply(self.std_devs_autoencoder3,
                                                                                       fill_value=0)
        df1 = pd.DataFrame({'Feature': weighted_sums1.index, 'Weighted_Sum': weighted_sums1.values})
        df2 = pd.DataFrame({'Feature': weighted_sums2.index, 'Weighted_Sum': weighted_sums2.values})
        df3 = pd.DataFrame({'Feature': weighted_sums3.index, 'Weighted_Sum': weighted_sums3.values})


        return df1, df2, df3




def load_fn(folder_path):
    fn1 = pd.read_csv(f'{folder_path}/1_featname.csv', header=None, names=['Feature'])['Feature'].tolist()
    fn2 = pd.read_csv(f'{folder_path}/2_featname.csv', header=None, names=['Feature'])['Feature'].tolist()
    fn3 = pd.read_csv(f'{folder_path}/3_featname.csv', header=None, names=['Feature'])['Feature'].tolist()
    omic1_data = pd.read_csv(f'{folder_path}/1_tr.csv', header=None)
    omic2_data = pd.read_csv(f'{folder_path}/2_tr.csv', header=None)
    omic3_data = pd.read_csv(f'{folder_path}/3_tr.csv', header=None)

    return [fn1, fn2, fn3], [omic1_data, omic2_data, omic3_data]


def calculate_std_devs(folder_path):
    feature_names, omic_data = load_fn(folder_path)
    std_devs_list = []

    for i, (fn, data) in enumerate(zip(feature_names, omic_data)):
        std_dev = data.std(axis=0)
        std_dev_df = pd.DataFrame({'Feature': fn, 'Standard Deviation': std_dev})
        std_dev_df['Omics'] = f'Omics_{i + 1}'
        std_devs_list.append(std_dev_df)

    return std_devs_list



