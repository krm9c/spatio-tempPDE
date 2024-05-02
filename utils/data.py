from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections import OrderedDict
import numpy as np
import torch
import torchvision
from PIL.Image import LANCZOS
# from torchmeta.transforms import Categorical, ClassSplitter, Rotation
import sklearn.model_selection as model_selection


from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


################################################
class Continual_Dataset(Dataset):
    def __init__(self, config, data_x, data_y):
        self.config = config
        self.x = data_x
        self.y = data_y
        # print(data_x,".", data_y, ".", self.x, self.y)
        if self.config['problem'] == 'classification':
            if self.config['network'] == 'fcnn':
                self.x = self.x.reshape([-1, 784])

    # A function to define the length of the problem
    def __len__(self):
        return self.x.shape[0]

    # A function to get samples
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.config['network'] == 'cnn':
            x_ = self.x[idx, :, :, :]
            y_ = self.y[idx]
        else:
            x_ = self.x[idx, :]
            y_ = self.y[idx]
        sample = (x_, y_)
        return sample

################################################


class data_return():
    def __init__(self, Config= { 'data_id': 'sine',
                                 'len_exp_replay':200 
                                }):
        self.dataset_id = Config['data_id']
        self.dataset = None
        self.len_exp_replay = Config['len_exp_replay']
        self.config=Config
        if self.dataset_id == 'omni':
            self.dataset = torchvision.datasets.Omniglot(
                root="../data", download=True, transform=transforms.Compose([
                    transforms.Resize(28, interpolation=LANCZOS),
                    transforms.ToTensor(),
                    lambda x: 1.0 - x,
                ]))
            [self.images, self.labels] = [list(t) for t in zip(*self.dataset)]
            self.images = torch.stack(self.images, dim=0)
            self.labels = np.array(self.labels)
            # print("The data shapes,", "[", self.images.shape,
            # self.labels.shape, "]")

        if self.dataset_id == 'mnist':
            my_transforms = transforms.Compose([
                transforms.ToTensor()])
            self.dataset = torchvision.datasets.MNIST('./data',
                                                      train=True, download=True, transform=my_transforms)
            [self.images, self.labels] = [list(t) for t in zip(*self.dataset)]
            self.images = torch.stack(self.images, dim=0)
            self.labels = np.array(self.labels)
            # print("The data shapes,", "[", self.images.shape,
            #       self.labels.shape, "]")
            # print("MNIST data")

        if self.dataset_id == 'sine':
            import pickle
            with open('../CL__jax/Incremental_Sine1e^3.p', 'rb') as fp:
                self.dataset = pickle.load(fp)
                # print("self dataset", self.dataset.keys())
        if self.dataset_id == 'synthetic':
            import pickle
            with open('../CL__jax/synthetic.p', 'rb') as fp:
                self.dataset = pickle.load(fp)


        self.y_test = None
        self.X_test = None
        self.y_train = None
        self.X_train = None
        self.exp_x_train = []
        self.exp_y_train = []
        self.exp_x_test = []
        self.exp_y_test = []

###############################################
    # This is the omniglot dataset function.
    def omni(self, task_id):
        task_id = int(task_id)
        idx = self.labels == task_id
        X = self.images[idx]
        y = self.labels[idx]
        # Split the data
        index = np.random.randint(0, X.shape[0], int(0.8*X.shape[0]))
        self.X_train = X[index, :]
        self.y_train = y[index]
        index = np.random.randint(0, X.shape[0], int(0.2*X.shape[0]))
        self.X_test = X[index, :]
        self.y_test = y[index]

###############################################
    def mnist(self, task_id):
        imp = np.random.randint(0, 9)
        # print(imp, task_id)
        idx = self.labels == imp
        X = self.images[idx]
        y = self.labels[idx]
        # print("We have to apply the transformation now.")
        rot_angle = np.random.random()*180
        scaling   = np.random.random()+1
        # print(rot_angle)
        X =torchvision.transforms.functional.affine(X, rot_angle,\
            translate = (scaling, scaling),\
            scale = 1, shear=rot_angle)
        # print("Just after the data is defined", X.shape, y.shape)
        # Split the data
        index = np.random.randint(0, X.shape[0], int(0.8*X.shape[0]))
        self.X_train = X[index, :]
        self.y_train = y[index]
        index = np.random.randint(0, X.shape[0], int(0.2*X.shape[0]))
        self.X_test = X[index, :]
        self.y_test = y[index]
        # print("just before I return the mnist dataset", self.X_train.shape, self.X_test.shape)

###############################################
    # This is the sine dataset function
    def sine(self, task_id):
        y, time, phase, amplitude, frequency = self.dataset['task'+str(task_id)]
        X = np.concatenate([phase, amplitude.reshape([-1, 1]),
                            frequency.reshape([-1, 1])], axis=1)
        print("checking shape for th sine", X.shape, y.shape)
        self.X_train, self.X_test,  self.y_train,  self.y_test \
            = model_selection.train_test_split(X, y, test_size=0.2)

###############################################
    def wind(self, task_id):
        # tid = np.random.randint(0,7)+1
        X, y = self.dataset['task'+str(task_id)]
        # print(y.shape, X.shape)
        # X = X + np.random.normal(sizes = (X.shape[0], 1))
        # print(X.shape, y.shape)
        self.X_train, self.X_test, self.y_train, self.y_test \
            = model_selection.train_test_split(X, y, test_size=0.2)


##############################################
    def append_to_experience(self, task_id):
        # Check if the arrays looks OK.
        if isinstance(self.X_train, np.ndarray):
            # print("how does this look")
            self.X_train = torch.from_numpy(self.X_train)
            self.X_test = torch.from_numpy(self.X_test)

        if task_id > 0:
            self.exp_x_test = torch.cat((self.exp_x_test, self.X_test), dim=0)
            self.exp_x_train = torch.cat((self.exp_x_train, self.X_train), dim=0)
            self.exp_y_test = np.concatenate([self.exp_y_test, self.y_test], axis=0)
            self.exp_y_train = np.concatenate(
                [self.exp_y_train, self.y_train], axis=0)
            # print("the experiance test shapes", self.exp_y_test.shape, self.exp_x_test.shape)
        else:
            self.exp_x_train.extend(self.X_train)
            self.exp_y_train.extend(self.y_train)
            self.exp_x_test.extend(self.X_test)
            self.exp_y_test.extend(self.y_test)

            # Convert the list into torch tensor
            # print("after extending", len(self.exp_x_train), len(self.exp_x_test))
            self.exp_x_train = torch.vstack(self.exp_x_train)
            self.exp_y_train = np.array(self.exp_y_train)
            self.exp_x_test = torch.vstack(self.exp_x_test)
            self.exp_y_test = np.array(self.exp_y_test)

        # Check for the length of the replay
        if len(self.exp_x_train) > self.config['len_exp_replay']:
            index = np.random.randint(
                0, self.exp_x_train.shape[0], self.config['len_exp_replay'])
            self.exp_x_train = self.exp_x_train[index, :]
            self.exp_y_train = self.exp_y_train[index]


        if len(self.exp_x_test) > self.config['len_exp_replay']:
            index = np.random.randint(
                0, self.exp_x_test.shape[0], self.config['len_exp_replay'])
            self.exp_x_test = self.exp_x_test[index, :]
            self.exp_y_test = self.exp_y_test[index]

    
    def retreive_data(self, task_id, phase):
        if phase == 'training':
            if task_id > 0:
                return (self.X_train, self.y_train), (self.exp_x_train, self.exp_y_train)
            else:
                # print("The shapes I am returning are -- task 0",  self.X_train.shape)
                return (self.X_train, self.y_train), (self.X_train, self.y_train)

        elif phase == 'testing':
            if task_id > 0:
                if self.config['data_id'] == 'omni':
                    index = np.random.randint(0, self.X_test.shape[0], 256)
                    # print("In the test function", self.X_test.shape, self.y_test.shape)
                    return (self.X_test[index, :], self.y_test[index]),\
                        (self.exp_x_test,self.exp_y_test)
                # print("In the test function", self.X_test.shape, self.y_test.shape)
                return (self.X_test, self.y_test), (self.exp_x_test, self.exp_y_test)
            else:
                if self.config['data_id'] == 'omni':
                    index = np.random.randint(0, self.X_test.shape[0], 256)
                    # print("In the test function", self.X_test.shape, self.y_test.shape)
                    return (self.X_test[index, :], self.y_test[index]), (self.X_test[index, :], self.y_test[index])
                return (self.X_test, self.y_test), (self.X_test, self.y_test)





    def generate_dataset(self, task_id, batch_size, phase):
        if phase == 'training':
            if self.dataset_id == 'omni':
                self.omni(task_id)
            elif self.dataset_id == 'mnist':
                self.mnist(task_id)
            elif self.dataset_id == 'sine':
                self.sine(task_id)
            elif self.dataset_id == 'wind':
                self.wind(task_id)

        (x, y), (dat_x, dat_y) = self.retreive_data(task_id, phase)
        print(x.shape, y.shape, len(dat_x), len(dat_y) )
        dataset_curr = Continual_Dataset(self.config, data_x=x, data_y=y)
        dataset_exp = Continual_Dataset(
            self.config, data_x=dat_x, data_y=dat_y)
        
        
        return DataLoader(dataset_curr, batch_size=self.config['batch_size'],
                          shuffle=True), \
            DataLoader(dataset_exp,  batch_size=self.config['batch_size'],
                       shuffle=True)




