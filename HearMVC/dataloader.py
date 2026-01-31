from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')
        print(self.data1.shape)
        print(self.data2.shape)
        print(self.data3.shape)
        # scipy.io.savemat('CCV.mat', {'X1': self.data1, 'X2': self.data2, 'X3': self.data3, 'Y': self.labels})

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class Caltech_6V(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)        
        scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(view):
        # for i in [0, 3]:
            self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class NUSWIDE(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        # scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        labels0 = data['Y'].T
        label_mapping = {14: 0, 19: 1, 23: 2, 28: 3, 29: 4}
        self.labels = np.vectorize(label_mapping.get)(labels0)
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        # print(self.class_num)
        # for i in range(5000):
        #     print(data['X1'][i][-1])
        # X1 = data['X1'][:, :-1]
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)][:, :-1].astype(np.float32))
            # self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)][:, :-1].shape)
            self.dims.append(data['X' + str(i + 1)][:, :-1].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class DHA(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)].astype(np.float32))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class YoutubeVideo(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        # scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        print(self.class_num)
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)].astype(np.float32))
            # self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])

        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class Caltech_5V(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path + 'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path + 'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.Y = labels
        self.xs = []
        self.xs.append(self.x1)
        self.xs.append(self.x2)

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.Y[idx]), torch.from_numpy(np.array(idx)).long()

class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000, )
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class msrcv1(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MSRCv1.mat')['Y'].astype(np.int32).reshape(210, )
        self.V1 = scipy.io.loadmat(path + 'MSRCv1.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MSRCv1.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'MSRCv1.mat')['X3'].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'MSRCv1.mat')['X4'].astype(np.float32)
        self.V5 = scipy.io.loadmat(path + 'MSRCv1.mat')['X5'].astype(np.float32)

    def __len__(self):
        return 210

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4),
                torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class scene(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'scene.mat')['Y'].astype(np.int32).reshape(2688, )
        self.V1 = scipy.io.loadmat(path + 'scene.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'scene.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'scene.mat')['X3'].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'scene.mat')['X4'].astype(np.float32)

    def __len__(self):
        return 2688

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3),torch.from_numpy(x4)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class handwritten(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'handwritten.mat')['Y'].astype(np.int32).reshape(2000, )
        self.V1 = scipy.io.loadmat(path + 'handwritten.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'handwritten.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'handwritten.mat')['X3'].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'handwritten.mat')['X4'].astype(np.float32)
        self.V5 = scipy.io.loadmat(path + 'handwritten.mat')['X5'].astype(np.float32)
        self.V6 = scipy.io.loadmat(path + 'handwritten.mat')['X6'].astype(np.float32)

    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3),torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Caltech":
        dataset = Caltech_6V('data/Caltech.mat', view=6)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "NUSWIDE":
        dataset = NUSWIDE('data/NUSWIDE.mat', view=5)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "DHA":
        dataset = DHA('data/DHA.mat', view=2)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "YoutubeVideo":
        dataset = YoutubeVideo("./data/Video-3V.mat", view=3)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "Caltech-2V":
        dataset = Caltech_5V('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech_5V('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech_5V('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech_5V('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "msrcv1":
        dataset = msrcv1('data/')
        # dims = [24, 576, 512, 256, 254]
        dims = [1302, 48, 512, 100, 256]  # 修改为真实维度
        view = 5
        class_num = 7
        data_size = 210
    elif dataset == "handwritten":
        dataset = handwritten('data/')
        dims = [240, 76, 216, 47, 64,6]
        view = 6
        class_num = 10
        data_size = 2000
    elif dataset == "scene":
        dataset = scene('data/')
        dims = [512, 432, 256,48]
        view = 4
        class_num = 8
        data_size = 2688
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
