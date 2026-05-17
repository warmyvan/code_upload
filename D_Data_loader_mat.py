from F_fct_module import *
import h5py
import scipy.io as sio
import os
from torch.utils.data import Dataset, DataLoader
from L_kl_g_loss import DiagonalGaussianDistribution



class MATLabelDataset(Dataset):

    def __init__(self, data_path, mat_files, type_label, label_file, key, num_samples):
        self.mat_files = mat_files
        self.key = key
        self.type_label = type_label
        self.num_samples = num_samples
        self.data_path = data_path
        self.label_file = label_file

        if type_label == 0:
            print('NO label')

        elif type_label == 1:
            print('descriptor label for FFT')
            with h5py.File(self.label_file, 'r') as dscrpt_data:
                '''
                Porosity 0 / Specific surface area 1 / Constrictivity 2 /Geodesic tortuosity 5 6 /Chord length 26 27 /Spherical contact distance 47 48 /
                '''
                dscrpt_ = dscrpt_data[self.key].astype(np.float32)  # (236,13500)
                dscrpt_ = np.vstack((dscrpt_[0], dscrpt_[1], dscrpt_[2],
                                     dscrpt_[5], dscrpt_[6],
                                     dscrpt_[26], dscrpt_[27],
                                     dscrpt_[47], dscrpt_[48],))  # (9,num_samples)
            self.dscrpt_ = dscrpt_

        elif type_label == 2:
            print('class label from mat_files name ')
            
        elif type_label == 22:
            print(' Slice Pore ')
        
        elif type_label == 33:
            print(' Slice Pore ')
            
        elif type_label == 44:
            print(' write slice pore ')
        
        elif type_label == 9:
            print("***free_layout*****")

        else:
            print(f'error for class MATLabelDataset param type_label:{type_label}')
            assert 0



    def __len__(self):
        return self.num_samples if self.num_samples<=len(self.mat_files) else len(self.mat_files)


    def __getitem__(self, idx):
        mat_file = self.data_path + self.mat_files[idx]
        mat_data = sio.loadmat(mat_file)
        data = mat_data[self.key]


        # data #
        if isinstance(data, np.ndarray) and data.dtype.names is not None:
            data_ = data['post'][0][0][0]
            data_ = sample_z(data_,seed=None)
            context_all = data['p'][0][0][0][0]
            context_x = data['px'][0][0][0]
            context_y = data['py'][0][0][0]
            context_z = data['pz'][0][0][0]
            raw_x = data['x'][0][0][0]

            if self.type_label == 0:
                label = torch.tensor(-1)
                return data_, label, context_all, context_x
            elif self.type_label == 9:
                label = torch.tensor(-1)
                return data_, label, context_all, raw_x
            elif self.type_label == 22:
                label = torch.tensor(-1)
                return data_, label, context_all, context_x, raw_x
            elif self.type_label == 33:
                label = torch.tensor(-1)
                return data_, label, context_all, context_x, context_y, context_z, raw_x
            elif self.type_label == 44:
                label = torch.tensor(-1)
                return data_, label, context_all, context_x, context_y, context_z, raw_x, self.mat_files[idx]
            else:
                assert 0            
        else:
            data = data.astype(np.float32)
            data = np.expand_dims(data, axis=0)
            data = torch.from_numpy(data)

            # label #
            if self.type_label == 0:
                label = torch.tensor(-1)
                return data, label

            elif self.type_label == 1:

                idx_dscrpt = int(self.mat_files[idx][10:-4]) - 1 

                dscrpt_ = self.dscrpt_[:, idx_dscrpt]
                dscrpt_ = dscrpt_.astype(np.float32)
                dscrpt_ = torch.from_numpy(dscrpt_)

                return data, dscrpt_

            elif self.type_label == 2:
                label = int(self.mat_files[idx][5:6])-1

                return data, label



def MAT_loader(*, if_train=True, data_train_path=None, data_test_path=None,
               type_label=2, label_path=None, mat_key='BW', num_sample=99999,
               batch_size=3, batch_size_test=1, **kwargs):
    print(f'data_train_path:{data_train_path}')
    if if_train == True:
        mat_files = [f for f in os.listdir(data_train_path) if f.endswith('.mat')]
        dataset = MATLabelDataset(data_train_path, mat_files, type_label, label_path,
                                  mat_key, num_sample)
        batch_size = batch_size
        print(f'len(dataset):{len(dataset)}')
        
    elif if_train == False:
        mat_files = [f for f in os.listdir(data_test_path) if f.endswith('.mat')]
        dataset = MATLabelDataset(data_test_path, mat_files, type_label, label_path,
                                  mat_key, num_sample)
        batch_size = batch_size_test
        print(f'len(dataset):{len(dataset)}')
    
    else:
        print(f"data loader false for def MAT_train_loader with param if_train :{if_train} ")
        assert 0==1


    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return data_loader
