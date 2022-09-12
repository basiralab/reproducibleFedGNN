import os
import torch
from dataLoader_medmnist import get_dl, get_G_list
from options import args_parser
import numpy as np
import cross_val
import random
from fed_localmodel import localModel
from options import args_parser
from  main_diffpool import model_diffpool, load_data
from  main_gcn import GCN
from learning_modes import federatedLearning, baseLine
from fed_accuracy import save_results
from fed_reproducibility import save_reproducibility
from Analysis import new_folder


def run(args, running_mode, dataset, model="", mode = "Federated"):
    """
    Parameters
    ----------
    args : Arguments
    running_mode : The purpose of the execution [Learn, Accuracy, Reproducibility]
        - Learn : Trains the locacl models
        - Accuracy : Plots the accuracy comaparison result
        - Reproducibility : Plots the heatmaps of reproducibility matrices
    dataset : Name of the dataset
    model : Name of the model
    mode : Learning mode [Federated, Baseline]
        - Federated : Trains local models with federated averaging
        - Baseline : Trains local models without federation
    Description
    ----------
    If the running mode is 'Learn', it creates local models with the given model and trains them according to given learning mode.
    If the running mode is 'Accuracy', it plots the accuracy comparison results of the given model. Before execution, make sure to train the given mode with both learning modes.
    If the running mode is 'Reproducibility', it plots the reproducibility matrices. Before execution, make sure to run all models with the given learning modes.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    hospital_num = args.hospital_num
    if running_mode == "Learn":
        if "ASDNC" in dataset or "Demo" in dataset:
            G_list = load_data(args)
            num_nodes = G_list[0]['adj'].shape[0]
        elif "MNIST" in dataset:
            train_loader = get_dl(dataset)
            G_list = get_G_list(train_loader, args)
            num_nodes = G_list[0]['adj'].shape[0]

        hospital_data = cross_val.stratify_splits(G_list, args)
        [random.shuffle(hospital_data[i]) for i in range(len(hospital_data))]

        global_model = None
        if model == "DiffPool":
            assign_input = num_nodes
            input_dim = num_nodes
            global_model = model_diffpool.SoftPoolingGcnEncoder(num_nodes, 
                                                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                                                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                                                bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                                                assign_input_dim=assign_input)
        elif model == "GCN":
            global_model = GCN(nfeat = num_nodes,
                                                nhid = args.hidden,
                                                nclass = args.num_classes,
                                                dropout = args.dropout)
        hospitals = []
        for h_num, data in enumerate(hospital_data):
            local_folds = cross_val.stratify_splits(data, args)
            hospitals.append(localModel(model, local_folds, args, len(local_folds) - 1, mode, local_name="hospital-" + str(h_num)+ "_"+ dataset))

        if not os.path.exists("./results/data"):
            os.makedirs("./results/data")
        
        new_folder(model.lower())
        if mode == "Federated":
            federatedLearning(global_model, hospitals, args, dataset, model)
        elif mode == "Baseline":
            baseLine(global_model, hospitals, args, dataset, model)
    elif running_mode == "Accuracy":
        '''
        Must run Federated and Baseline learning modes before saving the accuracy results
        '''
        save_results(model, dataset)
    elif running_mode == "Reproducibility":
        '''
        Must run all the models in given learning mode before saving the reproducibility results
        '''
        save_reproducibility(dataset, mode, hospital_num, args.input_type)

if __name__ == "__main__":
    args = args_parser()
    dataset = args.dataset
    running_modes = ["Learn", "Accuracy", "Reproducibility"]
    models = ["GCN", "DiffPool"]
    learning_modes = ["Federated", "Baseline"]

    
    
    for model in models:
        for mode in learning_modes:
            run(args,running_modes[0],dataset,model,mode)
        run(args,running_modes[1],dataset,model)

    for mode in learning_modes:
        run(args,running_modes[2],dataset,mode=mode)
