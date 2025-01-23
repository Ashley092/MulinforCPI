import time
import scipy
import random
import logging
from sklearn.metrics import roc_curve, auc, average_precision_score
import numpy as np
import scipy.stats
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def concordance_index(y_true, y_pred):
    """计算一致性指数 (CI)"""
    pair_count = 0
    concordant = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                pair_count += 1
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                        (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant += 1
    return concordant / pair_count if pair_count > 0 else 0


def calculate_metrics(y_true, y_pred):
    """计算所有评估指标"""
    metrics = {}

    # RMSE
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAE
    metrics['mae'] = mean_absolute_error(y_true, y_pred)

    # Pearson correlation
    metrics['pearson'], _ = pearsonr(y_true.squeeze(), y_pred.squeeze())

    # Spearman correlation
    metrics['spearman'], _ = spearmanr(y_true.squeeze(), y_pred.squeeze())

    # Concordance Index
    metrics['ci'] = concordance_index(y_true, y_pred)

    # R²m metric
    r2 = r2_score(y_true, y_pred)
    metrics['rm2'] = r2 * (1 - np.sqrt(abs(r2 - r2_score(y_true, y_pred, force_finite=True))))

    return metrics


def cal_affinity_torch(model, loader, batch_size):
    """计算亲和力预测和评估指标"""
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in loader:
            # 移动数据到设备
            batch_data = (prot_data.to(device), drug_data_ver.to(device),
                          drug_data_adj.to(device), prot_contacts.to(device))

            # 获取预测值
            affinity = model.forward_affinity(*batch_data)

            y_pred.extend(affinity.cpu().numpy())
            y_true.extend(label.cpu().numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # 计算所有指标
    metrics = calculate_metrics(y_true, y_pred)

    return metrics


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("../logs/log_{}_N_{:.3f}.txt".format(time.strftime("%Y-%m-%d %H-%M-%S"), np.random.uniform()))
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cal_affinity_torch(model, loader, batch_size):
    y_pred, labels = np.zeros(len(loader.dataset)), np.zeros(len(loader.dataset))

    batch = 0
    model.eval()
    for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in loader:
        prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.to(device), drug_data_ver.to(device), drug_data_adj.to(device), prot_contacts.to(device), prot_inter.to(device), prot_inter_exist.to(device), label.to(device)
        with torch.no_grad():
            _, affn = model.forward_inter_affn(prot_data, drug_data_ver, drug_data_adj, prot_contacts)

        if batch != len(loader.dataset) // batch_size:
            labels[batch*batch_size:(batch+1)*batch_size] = label.squeeze().cpu().numpy()
            y_pred[batch*batch_size:(batch+1)*batch_size] = affn.squeeze().detach().cpu().numpy()
        else:
            labels[batch*batch_size:] = label.squeeze().cpu().numpy()
            y_pred[batch*batch_size:] = affn.squeeze().detach().cpu().numpy()
        batch += 1

    mse = 0
    for n in range(labels.shape[0]):
        mse += (y_pred[n] - labels[n]) ** 2
    mse /= labels.shape[0]
    rmse = np.sqrt(mse)

    pearson, _ = scipy.stats.pearsonr(y_pred.squeeze(), labels.squeeze())
    # tau, _ = scipy.stats.kendalltau(y_pred.squeeze(), labels.squeeze())
    # rho, _ = scipy.stats.spearmanr(y_pred.squeeze(), labels.squeeze())

    return rmse, pearson


def cal_interaction_torch(model, loader, prot_length, comp_length, batch_size):
    outputs, labels, ind = np.zeros((len(loader.dataset), 1000, 56)), np.zeros((len(loader.dataset), 1000, 56)), np.zeros(len(loader.dataset))

    batch = 0
    model.eval()
    for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in loader:
        prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.to(device), drug_data_ver.to(device), drug_data_adj.to(device), prot_contacts.to(device), prot_inter.to(device), prot_inter_exist.to(device), label.to(device)
        with torch.no_grad():
            inter, _ = model.forward_inter_affn(prot_data, drug_data_ver, drug_data_adj, prot_contacts)

        if batch != len(loader.dataset) // batch_size:
            labels[batch*batch_size:(batch+1)*batch_size] = prot_inter.cpu().numpy()
            outputs[batch*batch_size:(batch+1)*batch_size] = inter.detach().cpu().numpy()
            ind[batch*batch_size:(batch+1)*batch_size] = prot_inter_exist.cpu().numpy()
        else:
            labels[batch*batch_size:] = prot_inter.cpu().numpy()
            outputs[batch*batch_size:] = inter.detach().cpu().numpy()
            ind[batch*batch_size:] = prot_inter_exist.cpu().numpy()
        batch += 1

    AP = []
    AUC = []
    # AP_margin = []
    # AUC_margin = []

    for i in range(labels.shape[0]):
        if ind[i] != 0:

            length_prot = int(prot_length[i])
            length_comp = int(comp_length[i])

            true_label_cut = np.asarray(labels[i])[:length_prot, :length_comp]
            true_label = np.reshape(true_label_cut, (length_prot*length_comp))
            full_matrix = np.asarray(outputs[i])[:length_prot, :length_comp]
            pred_label = np.reshape(full_matrix, (length_prot*length_comp))

            average_precision_whole = average_precision_score(true_label, pred_label)
            AP.append(average_precision_whole)

            fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label)
            roc_auc_whole = auc(fpr_whole, tpr_whole)
            AUC.append(roc_auc_whole)

            # true_label = np.amax(true_label_cut, axis=1)
            # pred_label = np.amax(full_matrix, axis=1)

            # average_precision_whole = average_precision_score(true_label, pred_label)
            # AP_margin.append(average_precision_whole)

            # fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label)
            # roc_auc_whole = auc(fpr_whole, tpr_whole)
            # AUC_margin.append(roc_auc_whole)

    return np.mean(AP), np.mean(AUC)



