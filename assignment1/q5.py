import numpy as np
import matplotlib.pyplot as plt
plt.tight_layout()
import solution
import random
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
import torch.optim as optim

MA01391 = np.array([
    [    87,    167,    281,     56,      8,    744,     40,    107,    851,      5,    333,     54,     12,     56,    104,    372,     82,    117,    402],
    [   291,    145,     49,    800,    903,     13,    528,    433,     11,      0,      3,     12,      0,      8,    733,     13,    482,    322,    181],
    [    76,    414,    449,     21,      0,     65,    334,     48,     32,    903,    566,    504,    890,    775,      5,    507,    307,     73,    266],
    [   459,    187,    134,     36,      2,     91,     11,    324,     18,      3,      9,    341,      8,     71,     67,     17,     37,    396,     59]
])

def main():
    normalized = MA01391 / MA01391.sum(axis=0, keepdims=1)
    plt.imshow(MA01391 / MA01391.sum(axis=0, keepdims=1), interpolation='nearest')
    plt.title("MA01391")
    plt.savefig("MA01391.png", bbox_inches='tight')
    plt.close()

    batch_size = 64
    learning_rate = 0.002

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set RNG
    seed = 42
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # investigate your data
    f = h5py.File('./assignment1/er.h5', 'r')
    f.keys()
    f.close()

    basset_dataset_test = solution.BassetDataset(path='./assignment1', f5name='er.h5', split='test')
    basset_dataloader_test = DataLoader(basset_dataset_test, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=12)

    #model = solution.Basset().to(device)
    model = torch.load("model_params.pt").to(device)

    model.eval()

    filters = model.conv1.weight


    max_vals = torch.Tensor(np.zeros(300)).cuda()
    with torch.no_grad():
        for batch in basset_dataloader_test:
            max_vals = torch.max(max_vals, torch.max(model.conv1(batch['sequence'].to(device)).permute(1, 0, 2, 3).flatten(1, 3), dim=1)[0])

    acc = torch.zeros((300, 19, 4))
    c = torch.nn.Conv2d(1, 300, (19, 4), stride=(1, 1))
    c.weight = filters
    c.bias = model.conv1.bias
    with torch.no_grad():
        for batch in basset_dataloader_test:
            out = torch.nn.functional.unfold(batch['sequence'].to(device), (19, 4), stride=(1, 1), padding=(9, 0))
            acc += torch.sum(out.reshape((64 * 600, 1, 19, 4)).expand(-1, 300, -1, -1) * (c(out.reshape((64 * 600, 1, 19, 4))).squeeze() > max_vals/2).unsqueeze(2).unsqueeze(3).expand(-1, -1, 19, 4), dim=0).cpu()

    acc = acc.numpy() / acc.numpy().sum(axis=1, keepdims=1)

    corr = np.zeros(300)
    normalized = normalized.transpose()
    for i in range(acc.shape[0]):
        corr[i] = np.corrcoef(normalized.flatten().tolist(), acc[i, :, :].flatten().tolist())[0, 1]
        plt.title(round(corr[i], 2))
        plt.imshow(acc[i].transpose(), interpolation='nearest')
        plt.savefig(f"imgs/{corr[i]}_{i}.png", bbox_inches='tight')
        plt.close()
    

if __name__ == '__main__':
    main()