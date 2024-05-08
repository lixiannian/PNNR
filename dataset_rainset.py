import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
import settings_rainset


class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings_rainset.data_dir, name)   
        self.clear_dir = os.path.join(settings_rainset.data_dir, 'GT')
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = settings_rainset.patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        O = cv2.imread(img_file).astype(np.float32) / 255
        B = cv2.imread(os.path.join(self.clear_dir, file_name)).astype(np.float32) / 255


        if settings_rainset.aug_data:
            O, B = self.crop(O, B, aug=True)
            O, B = self.flip(O, B)
            O, B = self.rotate(O, B)
        else:
            O, B = self.crop(O, B, aug=False)

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        sample = {'O': O, 'B': B}

        return sample

    def crop(self, O, B, aug=False):
        patch_size = self.patch_size
        h, w, c = O.shape

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi =   1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = O[r: r+p_h, c: c+p_w]
        B = B[r: r+p_h, c: c+p_w]

        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))

        return O, B

    def flip(self, O, B):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)

        return O, B

    def rotate(self, O, B):
        angle = self.rand_state.randint(-30, 30)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        return O, B


class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings_rainset.test_data_dir, name)
        self.clear_dir = os.path.join(settings_rainset.test_data_dir, 'GT')
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = settings_rainset.patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        O = cv2.imread(img_file).astype(np.float32) / 255
        name = file_name
        B = cv2.imread(os.path.join(self.clear_dir, name)).astype(np.float32) / 255

        # h, w, c = O.shape
        # p_h, p_w = self.patch_size, self.patch_size
        # r = self.rand_state.randint(0, h - p_h)
        # c = self.rand_state.randint(0, w - p_w)
        # O = O[r: r + p_h, c: c + p_w]
        # B = B[r: r + p_h, c: c + p_w]

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        sample = {'O': O, 'B': B, 'name': name}
        return sample


class TestRealDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings_rainset.data_dir)
        self.mat_files = os.listdir(self.root_dir)
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        O = cv2.imread(img_file).astype(np.float32) / 255
        # h, w, c = O.shape
        # p_h, p_w = self.patch_size, self.patch_size
        # r = self.rand_state.randint(0, h - p_h)
        # c = self.rand_state.randint(0, w - p_w)
        # O = O[r: r + p_h, c: c + p_w]
        # B = B[r: r + p_h, c: c + p_w]

        O = np.transpose(O, (2, 0, 1))
        sample = {'O': O, 'B': O, 'name': file_name}
        return sample


class ShowDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings_rainset.data_dir, name)
        self.img_files = sorted(os.listdir(self.root_dir))
        self.file_num = len(self.img_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.img_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        print(img_file)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255

        h, ww, c = img_pair.shape
        w = int(ww / 2)

        #h_8 = h % 8
        #w_8 = w % 8

        if settings_rainset.pic_is_pair:
            O = np.transpose(img_pair[:, w:], (2, 0, 1))
            B = np.transpose(img_pair[:, :w], (2, 0, 1))
        else:
            O = np.transpose(img_pair[:, :], (2, 0, 1))
            B = np.transpose(img_pair[:, :], (2, 0, 1))

        sample = {'O': O, 'B': B,'file_name':file_name[:-4]}

        return sample


if __name__ == '__main__':
    # dt = TrainValDataset('18//rain')
    # print('TrainValDataset')
    # for i in range(len(dt)):
    #     smp = dt[i]
    #     # print(i)
    #     for k, v in smp.items():
    #         print(k, v.shape, v.dtype, v.mean())

    dt = TestDataset('19//rain')
    print('TestDataset')
    for i in range(len(dt)):
        smp = dt[i]
        print(i)
        # for k, v in smp.items():
        #     print(k, v.shape, v.dtype, v.mean())

    # print()
    # print('ShowDataset')
    # dt = ShowDataset('test')
    # for i in range(10):
    #     smp = dt[i]
    #     for k, v in smp.items():
    #         print(k, v.shape, v.dtype, v.mean())
