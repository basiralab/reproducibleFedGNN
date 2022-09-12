import matplotlib.pyplot as plt
import numpy as np
import os
def save_results(model, dataset):
    """
    Parameters
    ----------
    model : model name (DiffPool or GCN)
    dataset : dataset
    
    Description
    ----------
    Plots and saves:
    Bar chart for accuracy comparison of Baseline and Fedarated
    Averaged hospital losses
    Hospital-specific losses
    """  
    fed_acc = np.load("./results/data/"+ dataset + "_" + model + "_fed_acc.npy") * 100
    fed_loss = np.load("./results/data/"+ dataset + "_" + model + "_fed_loss.npy")

    base_acc = np.load("./results/data/"+ dataset + "_" + model + "_base_acc.npy") * 100
    base_loss = np.load("./results/data/"+ dataset + "_" + model + "_base_loss.npy")

    fed_hospital_loss = np.load("./results/data/"+ dataset + "_" + model + "_fed_hospital_loss.npy")
    base_hospital_loss = np.load("./results/data/"+ dataset + "_" + model + "_base_hospital_loss.npy")

    fed_hospital_loss_mean = np.mean(fed_hospital_loss, axis=0)
    base_hospital_loss_mean = np.mean(base_hospital_loss, axis=0)
    
    fed_acc_mean = np.mean(fed_acc)
    base_acc_mean = np.mean(base_acc)

    fed_acc_std = np.std(fed_acc)
    base_acc_std = np.std(base_acc)

    modes = ["Baseline", "Federated"]
    x_pos = np.arange(len(modes))
    data = [base_acc_mean, fed_acc_mean]
    err = [fed_acc_std, base_acc_std]
    min_val = np.min([np.min(fed_acc), np.min(base_acc)])
    max_val = np.max([np.max(fed_acc) + 1, np.max(base_acc) + 1])
    fig, ax = plt.subplots()
    ax.bar(x_pos, data, yerr=err, align='center', alpha=0.5, ecolor='red',capsize=5)
    ax.set_ylabel("Validation Accuracy")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(modes)
    ax.set_title("Accuracy Comparison Baseline-Federated (" + model + ")")
    ax.set(ylim=[min_val, max_val])
    ax.yaxis.grid(True)

    for i, v in enumerate(data):
        plt.text(x_pos[i] - 0.3, v + 0.01, "acc: {:.2f}".format(v))
        plt.text(x_pos[i] + 0.05, v+ 0.01, "std: {:.2f}".format(err[i]))
    
    if not os.path.exists("./results/loss"):
        os.makedirs("./results/loss")
    if not os.path.exists("./results/acc"):
        os.makedirs("./results/acc")

    plt.tight_layout()
    plt.savefig("./results/acc/"+ dataset + "_"+ model +"_acc_comparison.png", dpi=900)
    plt.show()
    plt.close()
    plt.figure()
    plt.title("Training Losses of Hospitals (Federated)")
    plt.plot(fed_hospital_loss[0])
    plt.plot(fed_hospital_loss[1], 'r')
    plt.plot(fed_hospital_loss[2], 'black')
    plt.legend(["Hospital-1", "Hospital-2", "Hospital-3"], loc="upper right")
    plt.ylabel("Loss")
    plt.xlabel("100 Epochs Accross 5 rounds")
    plt.savefig("./results/loss/" + dataset + "_" + model + "_fed_hospital_loss.png")
    plt.close()
    plt.figure()
    plt.title("Training Losses of Hospitals (Baseline)")
    plt.plot(base_hospital_loss[0])
    plt.plot(base_hospital_loss[1], 'r')
    plt.plot(base_hospital_loss[2], 'black')
    plt.legend(["Hospital-1", "Hospital-2", "Hospital-3"], loc="upper right")
    plt.ylabel("Loss")
    plt.xlabel("100 Epochs Accross 5 rounds")
    plt.savefig("./results/loss/" + dataset + "_" + model + "_base_hospital_loss.png")
    plt.close()
    plt.figure()
    plt.title("Average Training Losses of Hospitals (Federated vs Baseline)")
    plt.plot(fed_hospital_loss_mean)
    plt.plot(base_hospital_loss_mean, 'black')
    plt.legend(["Federated","Baseline"], loc="upper right")
    plt.ylabel("Loss")
    plt.xlabel("100 Epochs Accross 5 rounds")
    plt.savefig("./results/loss/" + dataset + "_" + model + "_avg_base_vs_fed_loss.png")
    plt.close()