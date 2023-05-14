import pandas as pd
import numpy as np
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
import argparse
from mlp import MLPEngine
from data import SampleGenerator
from utils import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='PFedRec')
parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
parser.add_argument('--clients_sample_num', type=int, default=0)
parser.add_argument('--num_round', type=int, default=100)
parser.add_argument('--local_epoch', type=int, default=1)
parser.add_argument('--lr_eta', type=int, default=80)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--dataset', type=str, default='100k')
parser.add_argument('--num_users', type=int)
parser.add_argument('--num_items', type=int)
parser.add_argument('--latent_dim', type=int, default=32)
parser.add_argument('--num_negative', type=int, default=4)
parser.add_argument('--l2_regularization', type=float, default=0.)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--device_id', type=int, default=1)
parser.add_argument('--model_dir', type=str, default='checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model')
parser.add_argument('--ind', type=int, default=0)
args = parser.parse_args()

# Model.
config = vars(args)
if config['dataset'] == 'ml-1m':
    config['num_users'] = 6040
    config['num_items'] = 3706
elif config['dataset'] == '100k':
    config['num_users'] = 943
    config['num_items'] = 1682
elif config['dataset'] == 'lastfm-2k':
    config['num_users'] = 1600
    config['num_items'] = 12454
elif config['dataset'] == 'amazon':
    config['num_users'] = 8072
    config['num_items'] = 11830
else:
    pass
engine = MLPEngine(config)

# Logging.
path = 'log/'
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logname = os.path.join(path, current_time+'.txt')
initLogging(logname)

# Load Data
dataset_dir = "/data/" + config['dataset'] + "/" + "ratings.dat"
if config['dataset'] == "ml-1m":
    rating = pd.read_csv(dataset_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
elif config['dataset'] == "100k":
    rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
elif config['dataset'] == "lastfm-2k":
    rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
elif config['dataset'] == "amazon":
    rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    rating = rating.sort_values(by='uid', ascending=True)
else:
    pass
# Reindex
user_id = rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
rating = pd.merge(rating, user_id, on=['uid'], how='left')
item_id = rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
rating = pd.merge(rating, item_id, on=['mid'], how='left')
rating = rating[['userId', 'itemId', 'rating', 'timestamp']]
logging.info('Range of userId is [{}, {}]'.format(rating.userId.min(), rating.userId.max()))
logging.info('Range of itemId is [{}, {}]'.format(rating.itemId.min(), rating.itemId.max()))

# DataLoader for training
sample_generator = SampleGenerator(ratings=rating)
validate_data = sample_generator.validate_data
test_data = sample_generator.test_data

hit_ratio_list = []
ndcg_list = []
val_hr_list = []
val_ndcg_list = []
train_loss_list = []
test_loss_list = []
val_loss_list = []
best_val_hr = 0
final_test_round = 0
for round in range(config['num_round']):
    logging.info('-' * 80)
    logging.info('Round {} starts !'.format(round))

    all_train_data = sample_generator.store_all_train_data(config['num_negative'])
    logging.info('-' * 80)
    logging.info('Training phase!')
    tr_loss = engine.fed_train_a_round(all_train_data, round_id=round)
    # break
    train_loss_list.append(tr_loss)

    logging.info('-' * 80)
    logging.info('Testing phase!')
    hit_ratio, ndcg, te_loss = engine.fed_evaluate(test_data)
    test_loss_list.append(te_loss)
    # break
    logging.info('[Testing Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(round, hit_ratio, ndcg))
    hit_ratio_list.append(hit_ratio)
    ndcg_list.append(ndcg)

    logging.info('-' * 80)
    logging.info('Validating phase!')
    val_hit_ratio, val_ndcg, v_loss = engine.fed_evaluate(validate_data)
    val_loss_list.append(v_loss)
    logging.info(
        '[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(round, val_hit_ratio, val_ndcg))
    val_hr_list.append(val_hit_ratio)
    val_ndcg_list.append(val_ndcg)

    if val_hit_ratio >= best_val_hr:
        best_val_hr = val_hit_ratio
        final_test_round = round

current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
str = current_time + '-' + 'latent_dim: ' + str(config['latent_dim']) + '-' + 'lr: ' + str(config['lr']) + '-' + \
      'clients_sample_ratio: ' + str(config['clients_sample_ratio']) + '-' + 'num_round: ' + str(config['num_round']) \
      + '-' + 'negatives: ' + str(config['num_negative']) + '-' + 'lr_eta: ' + str(config['lr_eta']) + '-' + \
      'batch_size: ' + str(config['batch_size']) + '-' + 'hr: ' + str(hit_ratio_list[final_test_round]) + '-' \
      + 'ndcg: ' + str(ndcg_list[final_test_round]) + '-' + 'best_round: ' + str(final_test_round) + '-' + \
      'optimizer: ' + config['optimizer'] + '-' + 'l2_regularization: ' + str(config['l2_regularization'])
file_name = "sh_result/"+config['dataset']+".txt"
with open(file_name, 'a') as file:
    file.write(str + '\n')

logging.info('PFedRec')
logging.info('clients_sample_ratio: {}, lr_eta: {}, bz: {}, lr: {}, dataset: {}, factor: {}, negatives: {}'.
             format(config['clients_sample_ratio'], config['lr_eta'], config['batch_size'], config['lr'],
                    config['dataset'], config['latent_dim'], config['num_negative']))

logging.info('hit_list: {}'.format(hit_ratio_list))
logging.info('ndcg_list: {}'.format(ndcg_list))
logging.info('Best test hr: {}, ndcg: {} at round {}'.format(hit_ratio_list[final_test_round],
                                                             ndcg_list[final_test_round],
                                                             final_test_round))