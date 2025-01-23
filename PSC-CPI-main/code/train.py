import csv
import nni
import time
import argparse
import warnings
from utils import cal_affinity_torch, calculate_metrics
import torch

from model import *
from utils import *
from dataset import *

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--temp', type=float, default=0.5)
parser.add_argument('--intra_loss', type=float, default=1.0)

parser.add_argument('--task_mode', type=int, default=0)
parser.add_argument('--modality', type=str, default='seq_str_linear')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--seq_encoder', type=str, default='HRNN')
parser.add_argument('--str_encoder', type=str, default='GAT')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--train', type=int, default=1)

parser.add_argument('--aug_left', type=float, default=0.001)
parser.add_argument('--aug_right', type=float, default=0.999)

args = parser.parse_args()
param = args.__dict__
param.update(nni.get_next_parameter())


set_seed(param['seed'])

train_set = data_loader(name_split='train')
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=param['batch_size'], shuffle=True, pin_memory = True, num_workers=6)
val_set = data_loader(name_split='val')
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=param['batch_size'], shuffle=False, pin_memory = True, num_workers=6)

model = ProteinEmbed_Model(param).to(device)
optimizer = torch.optim.Adam(model.parameters(), float(param['learning_rate']))

fused_matrix = torch.tensor(np.load('../data/fused_matrix.npy')).to(device)
loss_val_best = 1e10
logger = get_root_logger()
es = 0

if param['train'] == 1:

    if param['pretrain'] == 1:
        pre_epochs = 100
    else:
        pre_epochs = 0

    for epoch in range(pre_epochs):
        model.train()
        loss_epoch, batch = 0, 0
        for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in train_loader:
            prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.to(device, non_blocking=True), drug_data_ver.to(device, non_blocking=True), drug_data_adj.to(device, non_blocking=True), prot_contacts.to(device, non_blocking=True), prot_inter.to(device, non_blocking=True), prot_inter_exist.to(device, non_blocking=True), label.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = model(prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label, fused_matrix, pre_train_mode='pre-train')
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step()

            # print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, batch, loss.detach().cpu().numpy()))
            loss_epoch += loss.detach().cpu().numpy()
            batch += 1

        if epoch % 10 == 0 or epoch <= 10:
            logger.warning('Pre-training Epoch: {}, Train loss: {}'.format(epoch, loss_epoch / batch))


    for epoch in range(param['epoch']):
        model.train()
        loss_epoch, batch = 0, 0
        for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in train_loader:
            prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.to(device, non_blocking=True), drug_data_ver.to(device, non_blocking=True), drug_data_adj.to(device, non_blocking=True), prot_contacts.to(device, non_blocking=True), prot_inter.to(device, non_blocking=True), prot_inter_exist.to(device, non_blocking=True), label.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = model(prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label, fused_matrix)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step()

            # print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, batch, loss.detach().cpu().numpy()))
            loss_epoch += loss.detach().cpu().numpy()
            batch += 1

        if es == 50:
            print("Early stopping!")
            break

        # 在训练循环中的验证部分
        model.eval()
        with torch.no_grad():
            val_metrics = cal_affinity_torch(model, val_loader, param['batch_size'])
            if epoch % 10 == 0 or epoch <= 10:
                logger.warning('Epoch: {}, Train loss: {:.4f} | Val RMSE: {:.4f} | Val Pearson: {:.4f}'.format(
                    epoch, loss_epoch / batch, val_metrics['rmse'], val_metrics['pearson']
                ))

            # 使用RMSE作为早停指标
            if val_metrics['rmse'] < loss_val_best:
                loss_val_best = val_metrics['rmse']
                model_state = model.state_dict()
                es = 0
            else:
                es += 1


del train_loader
del val_loader





outFile = open('../results/PerformMetrics.csv','a+', newline='')
writer = csv.writer(outFile, dialect='excel')
results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
for v, k in param.items():
    results.append(k)


model = ProteinEmbed_Model(param).to(device)
# model.load_state_dict(torch.load('../models/model_{}.pth'.format(param['seed'])))
model.load_state_dict(model_state)
model.eval()

rmse_sum = 0
auc_sum = 0


# 评估训练集
eval_set = data_loader('train')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=param['batch_size'],
                                        shuffle=False, pin_memory=True, num_workers=4)
metrics = cal_affinity_torch(model, eval_loader, param['batch_size'])
print("Train | RMSE: {:.4f} | MAE: {:.4f} | Pearson: {:.4f} | Spearman: {:.4f} | CI: {:.4f} | RM2: {:.4f}".format(
    metrics['rmse'], metrics['mae'], metrics['pearson'],
    metrics['spearman'], metrics['ci'], metrics['rm2']
))

# 评估验证集
eval_set = data_loader('val')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=param['batch_size'],
                                        shuffle=False, pin_memory=True, num_workers=6)
metrics = cal_affinity_torch(model, eval_loader, param['batch_size'])
print("Val | RMSE: {:.4f} | MAE: {:.4f} | Pearson: {:.4f} | Spearman: {:.4f} | CI: {:.4f} | RM2: {:.4f}".format(
    metrics['rmse'], metrics['mae'], metrics['pearson'],
    metrics['spearman'], metrics['ci'], metrics['rm2']
))

# 评估测试集
eval_set = data_loader('test')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=param['batch_size'],
                                        shuffle=False, pin_memory=True, num_workers=6)
metrics = cal_affinity_torch(model, eval_loader, param['batch_size'])
print("Test | RMSE: {:.4f} | MAE: {:.4f} | Pearson: {:.4f} | Spearman: {:.4f} | CI: {:.4f} | RM2: {:.4f}".format(
    metrics['rmse'], metrics['mae'], metrics['pearson'],
    metrics['spearman'], metrics['ci'], metrics['rm2']
))
results.extend([str(metrics[k]) for k in ['rmse', 'mae', 'pearson', 'spearman', 'ci', 'rm2']])
rmse_sum += metrics['rmse']




# Unseen protein evaluation
eval_set = data_loader('unseen_prot')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=param['batch_size'],
                                        shuffle=False, pin_memory=True, num_workers=6)
prot_length = np.load('../data/protein_uniq_prot_length.npy')
comp_length = np.load('../data/protein_uniq_comp_length.npy')
metrics = cal_affinity_torch(model, eval_loader, param['batch_size'])
print("Unseen Prot | RMSE: {:.4f} | MAE: {:.4f} | Pearson: {:.4f} | Spearman: {:.4f} | CI: {:.4f} | RM2: {:.4f}".format(
    metrics['rmse'], metrics['mae'], metrics['pearson'],
    metrics['spearman'], metrics['ci'], metrics['rm2']
))
results.extend([str(metrics[k]) for k in ['rmse', 'mae', 'pearson', 'spearman', 'ci', 'rm2']])
rmse_sum += metrics['rmse']




eval_set = data_loader('unseen_comp')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=param['batch_size'], shuffle=False, pin_memory = True, num_workers=6)
prot_length = np.load('../data/compound_uniq_prot_length.npy')
comp_length = np.load('../data/compound_uniq_comp_length.npy')
metrics = cal_affinity_torch(model, eval_loader, param['batch_size'])
print("Unseen Comp | RMSE: {:.4f} | MAE: {:.4f} | Pearson: {:.4f} | Spearman: {:.4f} | CI: {:.4f} | RM2: {:.4f}".format(
    metrics['rmse'], metrics['mae'], metrics['pearson'],
    metrics['spearman'], metrics['ci'], metrics['rm2']
))
results.extend([str(metrics[k]) for k in ['rmse', 'mae', 'pearson', 'spearman', 'ci', 'rm2']])
rmse_sum += metrics['rmse']



eval_set = data_loader('unseen_both')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=param['batch_size'], shuffle=False, pin_memory = True, num_workers=6)
prot_length = np.load('../data/double_uniq_prot_length.npy')
comp_length = np.load('../data/double_uniq_comp_length.npy')
metrics = cal_affinity_torch(model, eval_loader, param['batch_size'])
print("Unseen Both | RMSE: {:.4f} | MAE: {:.4f} | Pearson: {:.4f} | Spearman: {:.4f} | CI: {:.4f} | RM2: {:.4f}".format(
    metrics['rmse'], metrics['mae'], metrics['pearson'],
    metrics['spearman'], metrics['ci'], metrics['rm2']
))
results.extend([str(metrics[k]) for k in ['rmse', 'mae', 'pearson', 'spearman', 'ci', 'rm2']])
rmse_sum += metrics['rmse']




if param['task_mode'] == 0:
    final_rmse = rmse_sum/4.0
    results.append(str(final_rmse))
    nni.report_final_result(final_rmse)  # 使用RMSE作为优化目标
elif param['task_mode'] == 1:
    results.append(str(auc_sum/4.0))
    nni.report_final_result(-auc_sum/4.0)   
writer.writerow(results)