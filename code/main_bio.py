from utils import load_data2,load_data
from train_test_bio_n import train_and_evaluate,bio_feature,summarize_imp_feat
from Biomaker import load_fn,calculate_std_devs
from model import AemlpModel


if __name__ == "__main__":

    data_folder = '../data_5k/LGG1'
    model_folder = 'LGG'
    save_folder = 'Biomaker/LGG1'
    batch_size = 34
    num_epochs = 300
    ae_input = [2000, 2000, 548]
    ae_class = [2, 2, 2]
    input_dims = [180, 180, 180]
    num_classes = 2
    lambda_a =[208,1,1,1,197]
    mode = 1
    if mode == 1:
        train_loader, test_loader = load_data2(data_folder, batch_size)
        feature_names_lists, omic_data = load_fn(data_folder)
        std_devs_list = calculate_std_devs(data_folder)
        train_and_evaluate(train_loader, test_loader, feature_names_lists, std_devs_list, AemlpModel, num_epochs,
                           ae_input, ae_class, input_dims, num_classes,save_folder,lambda_a)
    if mode == 2:
        x_combined, y_combined, o1, o2, o3 = load_data(data_folder)
        data_list = [o1, o2, o3]
        view_list = [1,2,3]
        featimp_list = bio_feature(data_list, y_combined, model_folder, data_folder, view_list, AemlpModel, ae_input, ae_class,
                    input_dims, num_classes,lambda_a)
        summarize_imp_feat(featimp_list,save_folder)









