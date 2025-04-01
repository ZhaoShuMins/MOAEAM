import torch
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score,roc_auc_score
from utils import aemlp_loss
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from model import AemlpModel
from utils import load_data2
import os
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

    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
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


# In[57]:


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

def train_and_evaluate(train_loader,test_loader,AemlpModel, num_epochs=300, ae_input=[1000,1000,503], ae_class=[5,5,5], input_dims=[180,180,180],num_classes=5):

    for j in range(0, 1):  
        seed = 0
        seed_everything(seed)
        base_save_dir = 'model_save'

        hyperpm = parameter_parser()
        aemlp_model = AemlpModel(ae_input_dim=ae_input, ae_class_dim=ae_class, input_data_dims=input_dims, num_classes=num_classes, lambda_sa=79, lambda_or=45, lambda_reg=1, lambda_w=j+1,lambda_c=1,  hyperpm= hyperpm)
        aemlp_model.to(device)
        optimizer = torch.optim.Adam(aemlp_model.parameters(), lr=0.001)
        best_test_accuracy, b_weighted_f1, b_macro_f1, best_precision, best_recall = train_and_test_model(aemlp_model, train_loader, test_loader, optimizer, num_epochs=num_epochs, test_interval=10, patience=5, desired_performance=0.0, tolerance=0.01)



def train_and_evaluate_on_multiple_datasets(data_base_folder, datasets, batch_size, num_epochs, ae_input, ae_class,
                                            input_dims, num_classes,lambda_a):
    if num_classes == 2:
        results = {'accuracies': [], 'f1s': [], 'auc': [], 'precisions': [], 'recalls': []}
    else:
        results = {'accuracies': [], 'weighted_f1s': [], 'macro_f1s': [], 'precisions': [], 'recalls': []}

    for dataset in datasets:
        data_folder = os.path.join(data_base_folder, dataset)
        train_loader, test_loader = load_data2(data_folder, batch_size)  

        seed = 0
        seed_everything(seed)  

        hyperpm = parameter_parser()  

        
        aemlp_model = AemlpModel(ae_input_dim=ae_input, ae_class_dim=ae_class, input_data_dims=input_dims,
                                    num_classes=num_classes, lambda_sa=lambda_a[0], lambda_or=lambda_a[1], lambda_reg=lambda_a[2], lambda_w=lambda_a[3],
                                    lambda_c=lambda_a[4], hyperpm=hyperpm)
        aemlp_model.to(
            device)  

        optimizer = torch.optim.Adam(aemlp_model.parameters(), lr=0.001)

        if num_classes == 2:
            best_test_accuracy, b_f1, b_auc, best_precision, best_recall = train_and_test_model2(
                aemlp_model, train_loader, test_loader, optimizer, num_epochs=num_epochs, test_interval=10,
                patience=5,
                desired_performance=0.0, tolerance=0.01)

            results['accuracies'].append(best_test_accuracy)
            results['f1s'].append(b_f1)
            results['auc'].append(b_auc)
            results['precisions'].append(best_precision)
            results['recalls'].append(best_recall)
        else:
            best_test_accuracy, b_weighted_f1, b_macro_f1, best_precision, best_recall = train_and_test_model(
            aemlp_model, train_loader, test_loader, optimizer, num_epochs=num_epochs, test_interval=10, patience=5,
            desired_performance=0.0, tolerance=0.01)

            results['accuracies'].append(best_test_accuracy)
            results['weighted_f1s'].append(b_weighted_f1)
            results['macro_f1s'].append(b_macro_f1)
            results['precisions'].append(best_precision)
            results['recalls'].append(best_recall)
    
    def calculate_and_print_stats(results):
        for metric, values in results.items():
            if values:
                values_array = np.array(values)
                average = np.mean(values_array)
                std_dev = np.std(values_array)
                print(f"Average {metric}: {average:.4f}")
                print(f"Standard Deviation of {metric}: {std_dev:.4f}")
            else:
                print(f"No values for {metric}, skipping.")

    calculate_and_print_stats(results)
    
