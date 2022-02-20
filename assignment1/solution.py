# This script contains the helper functions you will be using for this assignment

import os
import random
from re import A

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BassetDataset(Dataset):
    """
    BassetDataset class taken with permission from Dr. Ahmad Pesaranghader

    We have already processed the data in HDF5 format: er.h5
    See https://www.h5py.org/ for details of the Python package used

    We used the same data processing pipeline as the paper.
    You can find the code here: https://github.com/davek44/Basset
    """

    # Initializes the BassetDataset
    def __init__(self, path='./data/', f5name='er.h5', split='train', transform=None):
        """
        Args:
            :param path: path to HDF5 file
            :param f5name: HDF5 file name
            :param split: split that we are interested to work with
            :param transform (callable, optional): Optional transform to be applied on a sample
        """

        self.split = split

        split_dict = {'train': ['train_in', 'train_out'],
                      'test': ['test_in', 'test_out'],
                      'valid': ['valid_in', 'valid_out']}

        assert self.split in split_dict, "'split' argument can be only defined as 'train', 'valid' or 'test'"

        # Open hdf5 file where one-hoted data are stored
        self.dataset = h5py.File(os.path.join(path, f5name.format(self.split)), 'r')

        # Keeping track of the names of the target labels
        self.target_labels = self.dataset['target_labels']

        # Get the list of volumes
        self.inputs = self.dataset[split_dict[split][0]]
        self.outputs = self.dataset[split_dict[split][1]]

        self.ids = list(range(len(self.inputs)))
        if self.split == 'test':
            self.id_vars = np.char.decode(self.dataset['test_headers'])

    def __getitem__(self, i):
        """
        Returns the sequence and the target at index i

        Notes:
        * The data is stored as float16, however, your model will expect float32.
          Do the type conversion here!
        * Pay attention to the output shape of the data.
          Change it to match what the model is expecting
          hint: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        * The target must also be converted to float32
        * When in doubt, look at the output of __getitem__ !
        """

        idx = self.ids[i]

        # Sequence & Target
        output = {'sequence': None, 'target': None}

        output['sequence'] = torch.Tensor(self.inputs[idx]).permute((1, 2, 0))#.to("cuda")
        output['target'] = torch.Tensor(self.outputs[idx])#.to("cuda")

        return output

    def __len__(self):
        return len(self.inputs)

    def get_seq_len(self):
        """
        Answer to Q1 part 2
        """

        return self.inputs.shape[-1]

    def is_equivalent(self):
        """
        Answer to Q1 part 3
        """
        
        return True


class Basset(nn.Module):
    """
    Basset model
    Architecture specifications can be found in the supplementary material
    You will also need to use some Convolution Arithmetic
    """

    def __init__(self):
        super(Basset, self).__init__()

        self.dropout = 0.3
        self.dropout_layer = nn.Dropout(self.dropout)
        self.num_cell_types = 164

        self.conv1 = nn.Conv2d(1, 300, (19, 4), stride=(1, 1), padding=(9, 0))
        self.conv2 = nn.Conv2d(300, 200, (11, 1), stride=(1, 1), padding=(5, 0))
        self.conv3 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(4, 0))

        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))

        self.fc1 = nn.Linear(2600, 1000)
        self.bn4 = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)

        self.fc3 = nn.Linear(1000, self.num_cell_types)

        #self.net.to("cuda")

    def forward(self, x):
        """
        Compute forward pass for the model.
        nn.Module will automatically create the `.backward` method!

        Note:
            * You will have to use torch's functional interface to 
              complete the forward method as it appears in the supplementary material
            * There are additional batch norm layers defined in `__init__`
              which you will want to use on your fully connected layers
            * Don't include the output activation here!
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.maxpool3(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = nn.functional.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        x = self.bn5(x)
        x = nn.functional.relu(x)
        x = self.dropout_layer(x)
        x = self.fc3(x)

        return x

def compute_fpr_tpr(y_true, y_pred):
    """
    Computes the False Positive Rate and True Positive Rate
    Args:
        :param y_true: groundtruth labels (np.array of ints [0 or 1])
        :param y_pred: model decisions (np.array of ints [0 or 1])

    :Return: dict with keys 'tpr', 'fpr'.
             values are floats
    """
    output = {'fpr': 0., 'tpr': 0.}

    output['tpr'] = np.sum(np.logical_and(y_pred == 1.0, y_true == 1.0)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
    output['fpr'] = np.sum(np.logical_and(y_pred == 1.0, y_true == 0.0)) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0 

    return output


def compute_fpr_tpr_dumb_model():
    """
    Simulates a dumb model and computes the False Positive Rate and True Positive Rate

    :Return: dict with keys 'tpr_list', 'fpr_list'.
             These lists contain the tpr and fpr for different thresholds (k)
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05
                 ...
                 output['fpr_list'][-1] corresponds to k=0.95

            Do the same for output['tpr_list']

    """
    output = {'fpr_list': [], 'tpr_list': []}

    y_target = np.random.random(size=1000) > 0.5
    y_pred = np.random.random(size=1000)

    for i in range(int(1 / 0.05)):
        fpr, tpr = list(compute_fpr_tpr(y_target, y_pred > (i * 0.05)).values())
        output['fpr_list'].append(fpr)
        output['tpr_list'].append(tpr)

    return output


def compute_fpr_tpr_smart_model():
    """
    Simulates a smart model and computes the False Positive Rate and True Positive Rate

    :Return: dict with keys 'tpr_list', 'fpr_list'.
             These lists contain the tpr and fpr for different thresholds (k)
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05
                 ...
                 output['fpr_list'][-1] corresponds to k=0.95

            Do the same for output['tpr_list']
    """

    output = {'fpr_list': [], 'tpr_list': []}

    y_target = np.random.random(size=1000) > 0.5
    y_pred = np.zeros(1000)
    y_pred[~y_target] = np.random.uniform(low=0.0, high=0.6, size=1000)[~y_target]
    y_pred[y_target] = np.random.uniform(low=0.4, high=1.0, size=1000)[y_target]

    for i in range(int(1 / 0.05)):
        fpr, tpr = list(compute_fpr_tpr(y_target, y_pred > (i * 0.05)).values())
        output['fpr_list'].append(fpr)
        output['tpr_list'].append(tpr)

    return output


def compute_auc_both_models():
    """
    Simulates a dumb model and a smart model and computes the AUC of both

    :Return: dict with keys 'auc_dumb_model', 'auc_smart_model'.
             These contain the AUC for both models
             auc values in the lists should be floats
    """
    output = {'auc_dumb_model': 0., 'auc_smart_model': 0.}

    fpr_tpr_dumb_model = compute_fpr_tpr_dumb_model()
    output['auc_dumb_model'] = sum([
        (fpr_tpr_dumb_model['fpr_list'][i] - fpr_tpr_dumb_model['fpr_list'][i + 1]) * fpr_tpr_dumb_model['tpr_list'][i]
        for i in range(len(fpr_tpr_dumb_model['fpr_list']) - 1)
    ])

    fpr_tpr_smart_model = compute_fpr_tpr_smart_model()
    output['auc_smart_model'] = sum([
        (fpr_tpr_smart_model['fpr_list'][i] - fpr_tpr_smart_model['fpr_list'][i + 1]) * fpr_tpr_smart_model['tpr_list'][i]
        for i in range(len(fpr_tpr_smart_model['fpr_list']) - 1)
    ])

    return output


def compute_auc_untrained_model(model, dataloader, device):
    """
    Computes the AUC of your input model

    Args:
        :param model: solution.Basset()
        :param dataloader: torch.utils.data.DataLoader
                           Where the dataset is solution.BassetDataset
        :param device: torch.device

    :Return: dict with key 'auc'.
             This contains the AUC for the model
             auc value should be float

    Notes:
    * Dont forget to re-apply your output activation!
    * Make sure this function works with arbitrarily small dataset sizes!
    * You should collect all the targets and model outputs and then compute AUC at the end
      (compute time should not be as much of a consideration here)
    """

    model.to(device)

    outputs, targets = [], []
    for batch in dataloader:
        out = model(batch['sequence'].to(device)).sigmoid().detach().cpu()
        target = batch['target'].detach().cpu()
        outputs.append(out)
        targets.append(target)

    return compute_auc(torch.cat(targets, dim=0).numpy(), torch.cat(outputs, dim=0).numpy())


def compute_auc(y_true, y_model):
    """
    Computes area under the ROC curve (using method described in main.ipynb)
    Args:
        :param y_true: groundtruth labels (np.array of ints [0 or 1])
        :param y_model: model outputs (np.array of float32 in [0, 1])
    :Return: dict with key 'auc'.
             This contains the AUC for the model
             auc value should be float

    Note: if you set y_model as the output of solution.Basset, 
    you need to transform it before passing it here!
    """
    output = {'auc': 0.}

    resolution = 100

    fpr_acc, tpr_acc = [], []
    for i in range(resolution):
        fpr, tpr = list(compute_fpr_tpr(y_true=y_true, y_pred=y_model > (i * (1 / resolution))).values())
        fpr_acc.append(fpr)
        tpr_acc.append(tpr)

    output['auc'] = sum([(fpr_acc[i] - fpr_acc[i + 1]) * tpr_acc[i] for i in range(resolution - 1)])

    return output


def get_critereon():
    """
    Picks the appropriate loss function for our task
    criterion should be subclass of torch.nn
    """

    critereon = torch.nn.BCEWithLogitsLoss()

    return critereon


def train_loop(model, train_dataloader, device, optimizer, criterion):
    """
    One Iteration across the training set
    Args:
        :param model: solution.Basset()
        :param train_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Return: total_score, total_loss.
             float of model score (AUC) and float of model loss for the entire loop (epoch)
             (if you want to display losses and/or scores within the loop, 
             you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!

    Note: you don’t need to compute the score after each training iteration.
    If you do this, your training loop will be really slow!
    You should instead compute it every 50 or so iterations and aggregate ...
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    for batch in train_dataloader:
        out = model(batch['sequence'])
        target = batch['target']
        
        loss = criterion(out, target)

        output['total_score'] += compute_auc(target.numpy(), out.sigmoid().detach().numpy())["auc"]
        output['total_loss'] += loss.detach()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return output['total_score'], output['total_loss']


def valid_loop(model, valid_dataloader, device, optimizer, criterion):
    """
    One Iteration across the validation set
    Args:
        :param model: solution.Basset()
        :param valid_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Return: total_score, total_loss.
             float of model score (AUC) and float of model loss for the entire loop (epoch)
             (if you want to display losses and/or scores within the loop, 
             you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!
    
    Note: if it is taking very long to run, 
    you may do simplifications like with the train_loop.
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    model.eval()

    targets, outputs = [], []
    with torch.no_grad():
        for batch in valid_dataloader:
            out = model(batch['sequence'])
            target = batch['target']
            outputs.append(out.sigmoid())
            targets.append(target)
            loss = criterion(out, target)

            output['total_loss'] += loss.detach()
        output['total_score'] = compute_auc(torch.cat(targets, dim=0).numpy(), torch.cat(outputs, dim=0).numpy())

    return output['total_score'], output['total_loss']

if __name__ == '__main__':
    import random
    import torch

    import numpy as np
    import h5py
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim

    import solution

    # The hyperparameters we will use
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

    basset_dataset_train = solution.BassetDataset(path='./assignment1', f5name='er.h5', split='train')
    basset_dataset_valid = solution.BassetDataset(path='./assignment1', f5name='er.h5', split='valid')
    basset_dataset_test = solution.BassetDataset(path='./assignment1', f5name='er.h5', split='test')
    basset_dataloader_train = DataLoader(basset_dataset_train,
                                        batch_size=batch_size,
                                        drop_last=True,
                                        shuffle=True,
                                        num_workers=4)
    basset_dataloader_valid = DataLoader(basset_dataset_valid,
                                        batch_size=batch_size,
                                        drop_last=True,
                                        shuffle=False,
                                        num_workers=4)
    basset_dataloader_test = DataLoader(basset_dataset_test,
                                        batch_size=batch_size,
                                        drop_last=True,
                                        shuffle=False,
                                        num_workers=4)

    basset_dataset_train.get_seq_len()

    model = solution.Basset()#.to(device)

    #solution.compute_fpr_tpr_dumb_model()

    #solution.compute_fpr_tpr_smart_model()

    #a = solution.compute_auc_both_models()

    #b = solution.compute_auc_untrained_model(model, basset_dataloader_test, device)

    criterion = solution.get_critereon()

    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate, betas=(0.9, 0.999))

    valid_score_best = 0
    patience = 2
    num_epochs = 5  # you don't need to train this for that long!

    for e in range(num_epochs):
        #train_score, train_loss = solution.train_loop(model, basset_dataloader_train, device, optimizer, criterion)
        valid_score, valid_loss = solution.valid_loop(model, basset_dataloader_valid, device, optimizer, criterion)

        print('epoch {}: loss={:.3f} score={:.3f}'.format(e,
                                                        valid_loss,
                                                        valid_score))

        if valid_score > valid_score_best:
            print('Best score: {}. Saving model...'.format(valid_score))
            torch.save(model, 'model_params.pt')
            valid_score_best = valid_score
        else:
            patience -= 1
            print('Score did not improve! {} <= {}. Patience left: {}'.format(valid_score,
                                                                            valid_score_best,
                                                                            patience))
        if patience == 0:
            print('patience reduced to 0. Training Finished.')
            break