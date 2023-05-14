import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import *
from metrics import MetronAtK
import random
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
from collections import OrderedDict


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        # self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        # self._writer.add_text('config', str(config), 0)
        self.server_model_param = {}
        self.client_model_params = {}
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()
        # mae metric
        self.mae = torch.nn.L1Loss()

    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def fed_train_single_batch(self, model_client, batch_data, optimizers):
        """train a batch and return an updated model."""
        # load batch data.
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()

        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()

        optimizer, optimizer_i = optimizers
        # update score function.
        optimizer.zero_grad()
        ratings_pred = model_client(items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        # update item embedding.
        optimizer_i.zero_grad()
        ratings_pred = model_client(items)
        loss_i = self.crit(ratings_pred.view(-1), ratings)
        loss_i.backward()
        optimizer_i.step()
        return model_client, loss_i.item()

    def aggregate_clients_params(self, round_user_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.
        t = 0
        for user in round_user_params.keys():
            # load a user's parameters.
            user_params = round_user_params[user]
            # print(user_params)
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data += user_params[key].data
            t += 1
        for key in self.server_model_param.keys():
            self.server_model_param[key].data = self.server_model_param[key].data / len(round_user_params)


    def fed_train_a_round(self, all_train_data, round_id):
        """train a round."""
        # sample users participating in single round.
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = random.sample(range(self.config['num_users']), num_participants)
        else:
            participants = random.sample(range(self.config['num_users']), self.config['clients_sample_num'])

        # store users' model parameters of current round.
        round_participant_params = {}
        # store all the users' train loss and mae.
        all_loss = {}

        # perform model update for each participated user.
        for user in participants:
            loss = 0
            # copy the client model architecture from self.model
            model_client = copy.deepcopy(self.model)
            # for the first round, client models copy initialized parameters directly.
            # for other rounds, client models receive updated item embedding and score function from server.
            if round_id != 0:
                user_param_dict = copy.deepcopy(self.model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                user_param_dict['embedding_item.weight'] = copy.deepcopy(self.server_model_param['embedding_item.weight'].data).cuda()
                model_client.load_state_dict(user_param_dict)
            # Defining optimizers
            # optimizer is responsible for updating score function.
            optimizer = torch.optim.SGD(model_client.affine_output.parameters(),
                                        lr=self.config['lr'], weight_decay=self.config['l2_regularization'])  # MLP optimizer
            # optimizer_i is responsible for updating item embedding.
            optimizer_i = torch.optim.SGD(model_client.embedding_item.parameters(),
                                          lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'],
                                          weight_decay=self.config['l2_regularization'])  # Item optimizer
            optimizers = [optimizer, optimizer_i]

            # load current user's training data and instance a train loader.
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            sample_num = 0
            # update client model.
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client, loss_u = self.fed_train_single_batch(model_client, batch, optimizers)
                    loss += loss_u * len(batch[0])
                    sample_num += len(batch[0])
                all_loss[user] = loss / sample_num
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # store client models' local parameters for personalization.
            self.client_model_params[user] = copy.deepcopy(client_param)
            for key in self.client_model_params[user].keys():
                self.client_model_params[user][key] = self.client_model_params[user][key].data.cpu()
            # store client models' local parameters for global update.
            round_participant_params[user] = copy.deepcopy(self.client_model_params[user])
            del round_participant_params[user]['affine_output.weight']
        # aggregate client models in server side.
        self.aggregate_clients_params(round_participant_params)
        return all_loss


    def fed_evaluate(self, evaluate_data):
        # evaluate all client models' performance using testing data.
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]
        # ratings for computing loss.
        temp = [0] * 100
        temp[0] = 1
        ratings = torch.FloatTensor(temp)
        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()
            ratings = ratings.cuda()
        # store all users' test item prediction score.
        test_scores = None
        # store all users' negative items prediction scores.
        negative_scores = None
        all_loss = {}
        for user in range(self.config['num_users']):
            # load each user's mlp parameters.
            user_model = copy.deepcopy(self.model)
            if user in self.client_model_params.keys():
                user_param_dict = copy.deepcopy(self.client_model_params[user])
                for key in user_param_dict.keys():
                    user_param_dict[key] = user_param_dict[key].data.cuda()
            else:
                user_param_dict = copy.deepcopy(self.model.state_dict())
            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            with torch.no_grad():
                # obtain user's positive test information.
                test_user = test_users[user: user + 1]
                test_item = test_items[user: user + 1]
                # obtain user's negative test information.
                negative_user = negative_users[user*99: (user+1)*99]
                negative_item = negative_items[user*99: (user+1)*99]
                # perform model prediction.
                test_score = user_model(test_item)
                negative_score = user_model(negative_item)
                if user == 0:
                    test_scores = test_score
                    negative_scores = negative_score
                else:
                    test_scores = torch.cat((test_scores, test_score))
                    negative_scores = torch.cat((negative_scores, negative_score))
                ratings_pred = torch.cat((test_score, negative_score))
                loss = self.crit(ratings_pred.view(-1), ratings)
            all_loss[user] = loss.item()
        if self.config['use_cuda'] is True:
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()
        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        return hit_ratio, ndcg, all_loss


    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)