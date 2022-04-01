import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def load_results(exp_dir, d, filename):
    values = []
    with open(os.path.join(exp_dir, d, filename)) as f:
        for l in f.readlines():
            values.append(float(l))

    return values

def main(exp_dir):
    exp = {}
    for d in os.listdir(exp_dir):
        exp[d] = {
            "train_loss": load_results(exp_dir, d, "debug/train_loss.txt"),
            "valid_loss": load_results(exp_dir, d, "debug/valid_loss.txt"),
            "train_ppl": load_results(exp_dir, d, "debug/train_ppl.txt"),
            "valid_ppl": load_results(exp_dir, d, "debug/valid_ppl.txt"),
            "train_time": load_results(exp_dir, d, "debug/train_time.txt"),
            "valid_time": load_results(exp_dir, d, "debug/valid_time.txt"),
        }

    # Plot loss
    for ex in exp.keys():
        plt.plot(range(1, len(exp[ex]["train_loss"]) + 1), exp[ex]["train_loss"], label=f"Training loss")
        plt.plot(range(1, len(exp[ex]["train_loss"]) + 1), exp[ex]["valid_loss"], label=f"Validation loss")
        plt.title(f"Training and validation losses for LSTM experiment with \n {ex.split('_')[0]} layer(s) and the {ex.split('_')[-1]} optimizer")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(np.arange(1, len(exp[ex]["train_loss"]) + 1))
        plt.legend(loc="upper right")
        plt.savefig(f"{exp_dir}/{ex}/debug/{ex}_loss.png")
        plt.close()

    # Plot perplexity
    for ex in exp.keys():
        plt.plot(range(1, len(exp[ex]["train_loss"]) + 1), exp[ex]["train_ppl"], label=f"Training perplexity")
        plt.plot(range(1, len(exp[ex]["train_loss"]) + 1), exp[ex]["valid_ppl"], label=f"Validation perplexity")
        plt.title(f"Training and validation perplexities for LSTM experiment with \n {ex.split('_')[0]} layer(s) and the {ex.split('_')[-1]} optimizer")
        plt.xlabel("Epoch")
        plt.ylabel("Perplexity")
        plt.xticks(np.arange(1, len(exp[ex]["train_loss"]) + 1))
        plt.legend(loc="upper right")
        plt.savefig(f"{exp_dir}/{ex}/debug/{ex}_perplexity.png")
        plt.close()

    ## Plot accuracy over 
    #for ex in exp.keys():
    #    plt.plot(range(1, len(exp[ex]["train_loss"]) + 1), exp[ex]["train_loss"], label=f"Training accuracy")
    #    plt.plot(range(1, len(exp[ex]["train_loss"]) + 1), exp[ex]["valid_loss"], label=f"Validation accuracy")
    #    plt.title(f"Training and validation accuracy over epochs for ViT experiment with \n {ex.split('_')[0]} layer(s) and the {' '.join(ex.split('_')[2:])} optimizer")
    #    plt.xlabel("Epoch")
    #    plt.ylabel("Accuracy")
    #    plt.xticks(np.arange(1, len(exp[ex]["train_loss"]) + 1))
    #    plt.legend(loc="lower right")
    #    plt.savefig(f"{exp_dir}/{ex}/debug/{ex}_accuracy_over_epoch.png")
    #    plt.close()
 #
    ## Plot perplexity
    #for ex in exp.keys():
    #    plt.plot([sum(exp[ex]["train_time"][0:i]) + (sum(exp[ex]["valid_time"][0:i-1]) if i else 0) for i in range(len(exp[ex]["train_time"]))], exp[ex]["train_ppl"], label=f"Training accuracy")
    #    plt.plot([sum(exp[ex]["train_time"][0:i]) + sum(exp[ex]["valid_time"][0:i]) for i in range(len(exp[ex]["train_time"]))], exp[ex]["valid_ppl"], label=f"Validation accuracy")
    #    plt.title(f"Training and validation accuracy over time for ViT experiment with \n {ex.split('_')[0]} layer(s) and the {' '.join(ex.split('_')[2:])} optimizer")
    #    plt.xlabel("Time")
    #    plt.ylabel("Accuracy")
    #    plt.legend(loc="lower right")
    #    plt.savefig(f"{exp_dir}/{ex}/debug/{ex}_accuracy_over_time.png")
    #    plt.close()

if __name__ == '__main__':
    main(sys.argv[1])