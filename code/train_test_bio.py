import torch
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score,roc_auc_score
from utils import aemlp_loss
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from model import AemlpModel
from Biomaker import biomaker,biomakera
import os
import pandas as pd
import argparse


def seed_everything(seed):

    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run")

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of conv layers.')
    parser.add_argument('--n_hidden', type=int, default=20,
                        help='Number of hidden units per modal.')
    parser.add_argument('--n_head', type=int, default=8,
                        help='Number of attention head.')
    parser.add_argument('--nmodal', type=int, default=3,
                        help='Number of omics.')

    return parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_test_model(model, train_loader, test_loader, optimizer, num_epochs, test_interval=None, patience=5,
                         desired_performance=0.0, tolerance=0.01):
    if test_interval is None:
        test_interval = num_epochs // 10 

    best_performance = desired_performance  
    b_weighted_f1 = None
    b_macro_f1 = None
    current_epoch = 0  
    wait = 0  

    min_epochs = 200  

    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()  

        for batch_o1, batch_o2, batch_o3, batch_y in train_loader:
            batch_o1, batch_o2, batch_o3, batch_y = batch_o1.to(device), batch_o2.to(device), batch_o3.to(device), batch_y.to(device)
            batch_inputs = (batch_o1, batch_o2, batch_o3)
          
            x_hat, y_sa, l1, l2, c3,x_c = model(batch_inputs)
    
            loss = aemlp_loss(model, batch_inputs, x_hat, y_sa, batch_y, l1, l2,c3,x_c, model.lambda_sa, model.lambda_or,
                              model.lambda_reg, model.lambda_w,model.lambda_c)
            if loss.dim() != 0:
                raise ValueError("Loss must be a scalar")
            total_loss += loss.item()
         
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % test_interval == 0 or epoch + 1 == num_epochs: 
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}')

            model.eval()  
            performance, accuracy, weighted_f1, macro_f1, precision, recall = test_model(model, test_loader)

            if accuracy > best_performance:
                best_performance = accuracy
                best_accuracy = accuracy
                best_test_loss = performance
                b_weighted_f1 = weighted_f1
                b_macro_f1 = macro_f1
                best_precision = precision
                best_recall = recall
                # save_path = os.path.join(save_dir, f'best_model_test_accuracy_{best_accuracy:.4f}_loss_{best_test_loss:.4f}.pth')
                # torch.save(model.state_dict(), save_path)   save_dir='model_save',
                # print(f'Model with best performance saved at {save_path}')

                wait = 0  
            else:
                wait += 1

            if epoch + 1 >= min_epochs and wait >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                print(f'Best_acc:{best_accuracy}')
                print(f'Best_f1_weight:{b_weighted_f1 }')
                print(f'Best_f1_marco:{b_macro_f1}')
                print(f'Best_precision:{best_precision}')
                print(f'best_recall:{best_recall}')
                break
    return best_accuracy, b_weighted_f1, b_macro_f1, best_precision, best_recall

def train_and_test_model2(model, train_loader, test_loader, optimizer, num_epochs, test_interval=None, patience=5,
                         desired_performance=0.0, tolerance=0.01):
    if test_interval is None:
        test_interval = num_epochs // 10 

    best_performance = desired_performance 
    best_test_accuracy = 0.0
    b_weighted_f1 = None
    b_macro_f1 = None
    current_epoch = 0  
    wait = 0  

    min_epochs = 200  

    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()

        for batch_o1, batch_o2, batch_o3, batch_y in train_loader:
            batch_o1, batch_o2, batch_o3, batch_y = batch_o1.to(device), batch_o2.to(device), batch_o3.to(device), batch_y.to(device)
            batch_inputs = (batch_o1, batch_o2, batch_o3)

            x_hat, y_sa, l1, l2, c3,x_c = model(batch_inputs)

            loss = aemlp_loss(model, batch_inputs, x_hat, y_sa, batch_y, l1, l2,c3,x_c, model.lambda_sa, model.lambda_or,
                              model.lambda_reg, model.lambda_w,model.lambda_c)
            if loss.dim() != 0:
                raise ValueError("Loss must be a scalar")
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % test_interval == 0 or epoch + 1 == num_epochs: 
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}')

            model.eval()  
            performance, accuracy, f1, precision, recall, auc = test_model2(model, test_loader)

            if accuracy > best_performance:
                best_performance = accuracy
                best_test_accuracy = accuracy
                best_test_f1 = f1
                best_precision = precision
                best_recall = recall
                best_auc = auc


                wait = 0
            else:
                wait += 1

            if epoch + 1 >= min_epochs and wait >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                print(f'Best_acc:{best_test_accuracy}')
                print(f'Best_f1:{best_test_f1 }')
                print(f'Best_auc:{best_auc}')
                print(f'Best_precision:{best_precision}')
                print(f'best_recall:{best_recall}')
                break
    return best_test_accuracy, best_test_f1, best_auc, best_precision, best_recall




def test_model(model, test_loader):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_o1, batch_o2, batch_o3, batch_y in test_loader:
            batch_o1, batch_o2, batch_o3, batch_y = batch_o1.to(device), batch_o2.to(device),batch_o3.to(device),batch_y.to(device)
            batch_inputs = (batch_o1, batch_o2, batch_o3)

            x_hat, y_sa, l1, l2, c3,x_c = model(batch_inputs)

            loss = aemlp_loss(model, batch_inputs, x_hat, y_sa, batch_y, l1, l2, c3,x_c, model.lambda_sa, model.lambda_or,
                              model.lambda_reg, model.lambda_w,model.lambda_c)
            total_loss += loss.item()


            _, predicted = torch.max(y_sa, 1)

            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())


    test_loss = total_loss / len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')


    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')


    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    return test_loss, accuracy, weighted_f1, macro_f1, precision, recall

def test_model2(model, test_loader):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    y_pred_proba_list = []

    with torch.no_grad():
        for batch_o1, batch_o2, batch_o3, batch_y in test_loader:
            batch_o1, batch_o2, batch_o3, batch_y = batch_o1.to(device), batch_o2.to(device),batch_o3.to(device),batch_y.to(device)
            batch_inputs = (batch_o1, batch_o2, batch_o3)

            x_hat, y_sa, l1, l2, c3,x_c = model(batch_inputs)

            loss = aemlp_loss(model, batch_inputs, x_hat, y_sa, batch_y, l1, l2, c3,x_c, model.lambda_sa, model.lambda_or,
                              model.lambda_reg, model.lambda_w,model.lambda_c)
            total_loss += loss.item()


            _, predicted = torch.max(y_sa, 1)

            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            y_pred_proba_list.append(y_sa.cpu().numpy())



    test_loss = total_loss / len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')


    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_pred_proba = np.concatenate(y_pred_proba_list, axis=0)

    test_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    return test_loss, accuracy, f1, precision, recall, auc


def train_and_evaluate(train_loader,test_loader,feature_names_lists,std_devs_list,AemlpModel, num_epochs, ae_input, ae_class, input_dims,num_classes,save_folder,lambda_a):

    seed = 0
    seed_everything(seed)

    hyperpm = parameter_parser()
    aemlp_model = AemlpModel(ae_input_dim=ae_input, ae_class_dim=ae_class, input_data_dims=input_dims, num_classes=num_classes, lambda_sa=lambda_a[0], lambda_or=lambda_a[1], lambda_reg=lambda_a[2], lambda_w=lambda_a[3],lambda_c=lambda_a[4],  hyperpm= hyperpm)
    aemlp_model.to(device)
    optimizer = torch.optim.Adam(aemlp_model.parameters(), lr=0.001)
    if num_classes == 2:
        train_and_test_model2(aemlp_model, train_loader, test_loader, optimizer, num_epochs=num_epochs, test_interval=10, patience=5, desired_performance=0.0, tolerance=0.01)
    else:
        train_and_test_model(aemlp_model, train_loader, test_loader, optimizer, num_epochs=num_epochs, test_interval=10, patience=5, desired_performance=0.0, tolerance=0.01)
    biomaker_instance = biomaker(feature_names_lists,std_devs_list)
    biomaker_instance_all = biomakera(feature_names_lists, std_devs_list)
    sorted_features_autoencoder1, sorted_features_autoencoder2, sorted_features_autoencoder3 = biomaker_instance(aemlp_model)
    df1,df2,df3 = biomaker_instance_all(aemlp_model)
    save_path1 = os.path.join(save_folder, 'df1.txt')
    save_path2 = os.path.join(save_folder, 'df2.txt')
    save_path3 = os.path.join(save_folder, 'df3.txt')
    df1.to_csv(save_path1, sep='\t', index=False, header=False)
    df2.to_csv(save_path2, sep='\t', index=False, header=False)
    df3.to_csv(save_path3, sep='\t', index=False, header=False)
    print("\nTop 10 features for Autoencoder 1:")
    print(sorted_features_autoencoder1)
    print("\nTop 10 features for Autoencoder 2:")
    print(sorted_features_autoencoder2)
    print("\nTop 10 features for Autoencoder 3:")
    print(sorted_features_autoencoder3)


def test_m(data_list, y_combined,model):
    model.eval()
    y_pred = []
    y_pred_proba_list = []
    y_true = []
    with torch.no_grad():
        o1_, o2_, o3_ = data_list[0],data_list[1],data_list[2]
        o1_, o2_, o3_,y_combined =o1_.to(device), o2_.to(device),o3_.to(device),y_combined.to(device)
        batch_inputs = (o1_, o2_, o3_)

        x_hat, y_sa, l1, l2, c3,x_c = model(batch_inputs)

        _, predicted = torch.max(y_sa, 1)

        y_true.extend(y_combined.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

        y_pred_proba_list.append(y_sa.cpu().numpy())
        f1 = f1_score(y_true, y_pred, average='macro')
    return f1


def bio_feature(data_list, y_combined,model_folder,data_folder,view_list,AemlpModel,ae_input, ae_class, input_dims,num_classes,lambda_a):
    seed = 0
    seed_everything(seed)
    model_path = os.path.join(model_folder, 'aemlp_model.pth')
    hyperpm = parameter_parser()
    aemlp_model = AemlpModel(ae_input_dim=ae_input, ae_class_dim=ae_class, input_data_dims=input_dims, num_classes=num_classes, lambda_sa=lambda_a[0], lambda_or=lambda_a[1], lambda_reg=lambda_a[2], lambda_w=lambda_a[3],lambda_c=lambda_a[4],  hyperpm= hyperpm)
    aemlp_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    aemlp_model.to(device)
    f1 = test_m(data_list, y_combined,aemlp_model)
    featname_list = []
    for v in view_list:
        df = pd.read_csv(os.path.join(data_folder, str(v)+"_featname.csv"), header=None)
        featname_list.append(df.values.flatten())
    feat_imp_list = []
    original_data_list = [d.clone() for d in data_list]
    for i, data in enumerate(data_list):
        num_features = data.shape[1]
        feat_imp = {"feat_name": featname_list[i]}
        feat_imp['imp'] = np.zeros(num_features)
        for j in range(num_features):
            data[:, j] = 0
            f1_tmp = test_m(data_list, y_combined, aemlp_model)
            data[:, j] = original_data_list[i][:, j]
            feat_imp['imp'][j] = f1 - f1_tmp
        feat_imp_list.append(pd.DataFrame(data=feat_imp))
    return feat_imp_list


def summarize_imp_feat(featimp_list,save_folder, topn=30):
    df_featimp = pd.concat(featimp_list, ignore_index=True)
    df_featimp_top = df_featimp.groupby('feat_name')['imp'].mean().reset_index()
    save_path = os.path.join(save_folder, 'df_featimp_all.txt')
    df_featimp_top.to_csv(save_path, sep='\t', index=False, header=True)
    df_featimp_top = df_featimp_top.sort_values(by='imp', ascending=False)
    df_featimp_top = df_featimp_top.iloc[:topn]

    print('{:}\t{:}'.format('Rank', 'Feature name'))
    for i in range(len(df_featimp_top)):
        print('{:}\t{:}'.format(i + 1, df_featimp_top.iloc[i]['feat_name']))
