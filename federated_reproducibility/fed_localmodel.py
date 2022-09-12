import torch
import numpy as np
import random
import main_diffpool
import main_gcn
import copy
import cross_val
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def average_weights(weights):
    """
    Parameters
    ----------
    weights : weights matrix of hospitals
    
    Description
    ----------
    This method applies the federated weight averaging and returns average of weights

    """
    w_avg = copy.deepcopy(weights[0])
    for key in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[key] += weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(weights))
    return w_avg

class localModel(object):
    def __init__(self, model_name, local_folds, args, val_idx,  mode = "Federated", local_name=""):
        self.model_name = model_name
        self.args = args
        self.args.local_ep = 50
        self.mode = mode
        self.local_name = local_name
        train, val, test = cross_val.datasets_splits(local_folds, args, val_idx)
        if self.args.evaluation_method =='model selection':
            self.train_dataset, self.val_dataset, self.threshold_value = cross_val.model_selection_split(train, val, args)
        if self.args.evaluation_method =='model assessment':
            self.train_dataset, self.val_dataset, self.threshold_value = cross_val.model_assessment_split(train, val, test, args)
    
    def localUpdate(self, model):
        """
        Parameters
        ----------
        model : Deep copy of the global model
        
        Description
        ----------
        This method updates the model locally and returns:
        locally learned weights
        test accuracy
        training loss

        """
        model.to(device)
        if self.model_name == "DiffPool":
            test_acc, train_loss = main_diffpool.train(self.args, self.train_dataset, self.val_dataset, model, self.threshold_value, self.mode + "_" + self.local_name)
        elif self.model_name == "GCN":
            test_acc, train_loss = main_gcn.train(self.args, self.train_dataset,self.val_dataset,model, self.threshold_value, self.mode + "_" + self.local_name)
        return model.state_dict(), test_acc, train_loss
                
