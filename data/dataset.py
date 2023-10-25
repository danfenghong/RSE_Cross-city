import torch
import torch.nn as nn
import torch.utils.data as Data
import slidingwindow as sw
import numpy as np
import scipy.io as sio
import os
import h5py
from sklearn.decomposition import PCA
from imgaug import augmenters as iaa


def getdata(dataset, patch, overlay, batchsize, pac_flag=False, band_norm_flag=False, aug_flag=False):
    label_train, label_valid, label_test, num_classes, band = slide_crop(dataset, patch, overlay, pac_flag,
                                                                             band_norm_flag, aug_flag)
    label_test_loader = Data.DataLoader(label_test, batch_size=1, shuffle=False, drop_last=False)
    label_train_loader = Data.DataLoader(label_train, batch_size=batchsize, shuffle=True, num_workers=0,
                                         pin_memory=True, drop_last=True)
    label_valid_loader = Data.DataLoader(label_test, batch_size=batchsize, shuffle=True, num_workers=0,
                                         pin_memory=True, drop_last=True)
    return label_train_loader, label_valid_loader, label_test_loader, num_classes, band


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def band_normalization(data):
    """ normalize the matrix to (0,1), r.s.t A axis (Default=0)
        return normalized matrix and a record matrix for normalize back
    """
    size = data.shape
    if len(size) != 3:
        raise ValueError("Unknown dataset")
    for i in range(size[-1]):
        _range = np.max(data[:, :, i]) - np.min(data[:, :, i])
        data[:, :, i] = (data[:, :, i] - np.min(data[:, :, i])) / _range
    return data


def read_data(dataset, pca_flag=False, band_norm=False):
    if dataset == 'augsburg':
        num_classes, band = 14, 242
        train_file = r'data/data1/augsburg_multimodal.mat'
        col_train, row_train = 1360, 886
        valid_file = r'data/data1/berlin_multimodal.mat'
        col_valid, row_valid = 811, 2465
        input_data = sio.loadmat(train_file)
        valid_data = sio.loadmat(valid_file)

        hsi = input_data['HSI']  # 886 1360 242
        hsi = hsi.astype(np.float32)
        msi = input_data['MSI']
        msi = msi.astype(np.float32)
        sar = input_data['SAR']
        sar = sar.astype(np.float32)
        label = input_data['label']

        hsi_valid = valid_data['HSI'][:, :, 0:band]  # 2456 811 242
        hsi_valid = hsi_valid.astype(np.float32)
        msi_valid = valid_data['MSI']
        msi_valid = msi_valid.astype(np.float32)
        sar_valid = valid_data['SAR']
        sar_valid = sar_valid.astype(np.float32)
        label_valid = valid_data['label']

        # a = np.min(label_valid)
        # PCA
        if pca_flag:
            hsi_matrix = np.reshape(hsi, (hsi.shape[0] * hsi.shape[1], hsi.shape[2]))  # 2456*811 242
            pca = PCA(n_components=10)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 10
            hsi = np.reshape(hsi_matrix, (hsi.shape[0], hsi.shape[1], pca.n_components_))

            hsi_matrix = np.reshape(hsi_valid,
                                    (hsi_valid.shape[0] * hsi_valid.shape[1], hsi_valid.shape[2]))  # 2456*811 242
            pca = PCA(n_components=10)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 10
            hsi_valid = np.reshape(hsi_matrix, (hsi_valid.shape[0], hsi_valid.shape[1], pca.n_components_))

            band = 10

            del hsi_matrix

    elif dataset == 'beijing':
        num_classes = 14
        band = 10  # 116
        ## beijing is training, wuhan is testing
        train_file = r'data/data2/beijing.mat'
        train_file_label = r'data/data2/beijing_label.mat'
        col_train, row_train = 13474, 8706
        valid_file = r'data/data2/wuhan.mat'
        valid_file_label = r'data/data2/wuhan_label.mat'
        col_valid, row_valid = 6225, 8670

        with h5py.File(train_file, 'r') as f:
            f = h5py.File(train_file, 'r')
        hsi = np.array(f['HSI'])
        msi = np.transpose(f['MSI'])
        sar = np.transpose(f['SAR'])
        # idx = np.where(np.isnan(sar))
        sar[1097, 8105, 1] = sum([sar[1096, 8105, 1], sar[1098, 8105, 1], sar[1097, 8104, 1], sar[1097, 8106, 1]]) / 4

        # cut beijing suburb region
        cut_length = 0
        col_train = col_train - cut_length
        hsi = hsi[:, :, cut_length // 3:]
        msi = msi[cut_length:, :, :]
        sar = sar[cut_length:, :, :]

        # applying PCA for HSI # hsi (116, 2903, 4492)
        hsi_matrix = np.reshape(np.transpose(hsi), (hsi.shape[1] * hsi.shape[2], hsi.shape[0]))  # 2903*4492 116
        pca = PCA(n_components=10)
        pca.fit_transform(hsi_matrix)
        newspace = pca.components_
        newspace = newspace.transpose()  # 116*10
        hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2903*4492 10
        hsi_cube = np.transpose(np.reshape(hsi_matrix, (hsi.shape[2], hsi.shape[1], pca.n_components_)))
        del hsi

        mm = nn.Upsample(scale_factor=3, mode='nearest', align_corners=None)
        # upsample from 30m to 10m
        hsi1 = mm(torch.from_numpy(hsi_cube).unsqueeze(0)).squeeze().numpy()
        hsi1 = np.transpose(hsi1)
        # remove extra pixels
        hsi = hsi1[:col_train, :row_train, :]
        del hsi1

        with h5py.File(train_file_label, 'r') as f:
            f = h5py.File(train_file_label, 'r')
        label = np.transpose(f['label'])
        # cut beijing label
        label = label[cut_length:, :]

        with h5py.File(valid_file, 'r') as f:
            f = h5py.File(valid_file, 'r')
        hsi_valid = np.array(f['HSI'])
        msi_valid = np.transpose(f['MSI'])
        sar_valid = np.transpose(f['SAR'])

        ## applying PCA for valid HSI
        hsi_matrix = np.reshape(np.transpose(hsi_valid), (hsi_valid.shape[1] * hsi_valid.shape[2], hsi_valid.shape[0]))
        pca = PCA(n_components=10)
        pca.fit_transform(hsi_matrix)
        newspace = pca.components_
        newspace = newspace.transpose()
        hsi_matrix = np.matmul(hsi_matrix, newspace)
        hsi_cube = np.transpose(np.reshape(hsi_matrix, (hsi_valid.shape[2], hsi_valid.shape[1], pca.n_components_)))
        del hsi_valid

        hsi1 = mm(torch.from_numpy(hsi_cube).unsqueeze(0)).squeeze().numpy()
        hsi_valid = np.transpose(hsi1)
        del hsi1

        with h5py.File(valid_file_label, 'r') as f:
            f = h5py.File(valid_file_label, 'r')
        label_valid = np.transpose(f['label'])

    else:
        raise ValueError("Unknown dataset")

    # normalize data
    if band_norm:
        norm = band_normalization
    else:
        norm = normalization

    hsi = norm(hsi)
    msi = norm(msi)
    sar = norm(sar)
    hsi_valid = norm(hsi_valid)
    msi_valid = norm(msi_valid)
    sar_valid = norm(sar_valid)

    return hsi, msi, sar, label, hsi_valid, msi_valid, sar_valid, label_valid, num_classes, band


def slide_crop(dataset, patch, overlay, pca_flag=False, band_norm_flag=False, aug_flag=False):
    hsi, msi, sar, label, hsi_valid, msi_valid, sar_valid, label_valid, num_classes, band = read_data(dataset, pca_flag,
                                                                                                      band_norm_flag)

    if dataset == 'augsburg':
        col_train, row_train = 1360, 886
        col_valid, row_valid = 811, 2465

    elif dataset == 'beijing':
        col_train, row_train = 8706, 13474
        col_valid, row_valid = 8670, 6225

    else:
        raise ValueError("Unknown dataset")

    transform = iaa.Sequential([
        iaa.Rot90([0, 1, 2, 3]),
        iaa.VerticalFlip(p=0.5),
        iaa.HorizontalFlip(p=0.5),
    ])

    # slide crop for train data
    window_set_train = sw.generate(hsi, sw.DimOrder.HeightWidthChannel, patch, overlay)
    hsi_list = []
    msi_list = []
    sar_list = []
    label_list = []
    for window in window_set_train:
        subset_hsi = hsi[window.indices()]
        subset_msi = msi[window.indices()]
        subset_sar = sar[window.indices()]
        subset_label = label[window.indices()]
        if aug_flag:
            all_img = np.concatenate((subset_hsi, subset_msi, subset_sar), axis=-1)
            img1, label1 = transform(image=all_img,
                                     segmentation_maps=np.stack(
                                         (subset_label[np.newaxis, :, :], subset_label[np.newaxis, :, :])
                                         , axis=-1).astype(np.int32))

            subset_label = label1[0, :, :, 0]
            subset_hsi = img1[:, :, :10]
            subset_msi = img1[:, :, 10:14]
            subset_sar = img1[:, :, 14:16]

        hsi_list.append(subset_hsi)
        msi_list.append(subset_msi)
        sar_list.append(subset_sar)
        label_list.append(subset_label)
    del hsi, msi, sar, label
    hsi_list = np.array(hsi_list)
    msi_list = np.array(msi_list)
    sar_list = np.array(sar_list)
    label_list = np.array(label_list)
    # has_zero = np.any(label_list==0)
    # print(has_zero)

    # non-overlay crop for valid data
    window_set_valid = sw.generate(hsi_valid, sw.DimOrder.HeightWidthChannel, patch, overlay)
    hsi_valid_list = []
    msi_valid_list = []
    sar_valid_list = []
    label_valid_list = []
    for window in window_set_valid:
        subset_hsi = hsi_valid[window.indices()]
        subset_msi = msi_valid[window.indices()]
        subset_sar = sar_valid[window.indices()]
        subset_label = label_valid[window.indices()]
        if aug_flag:
            all_img = np.concatenate((subset_hsi, subset_msi, subset_sar), axis=-1)
            img1, label1 = transform(image=all_img,
                                     segmentation_maps=np.stack(
                                         (subset_label[np.newaxis, :, :], subset_label[np.newaxis, :, :])
                                         , axis=-1).astype(np.int32))

            subset_label = label1[0, :, :, 0]
            subset_hsi = img1[:, :, :10]
            subset_msi = img1[:, :, 10:14]
            subset_sar = img1[:, :, 14:16]
        hsi_valid_list.append(subset_hsi)
        msi_valid_list.append(subset_msi)
        sar_valid_list.append(subset_sar)
        label_valid_list.append(subset_label)
    hsi_valid_list = np.array(hsi_valid_list)
    msi_valid_list = np.array(msi_valid_list)
    sar_valid_list = np.array(sar_valid_list)
    label_valid_list = np.array(label_valid_list)

    # non-overlay crop for valid data
    window_set_test = sw.generate(hsi_valid, sw.DimOrder.HeightWidthChannel, patch, 0)
    hsi_test_list = []
    msi_test_list = []
    sar_test_list = []
    label_test_list = []
    for window in window_set_test:
        subset_hsi = hsi_valid[window.indices()]
        subset_msi = msi_valid[window.indices()]
        subset_sar = sar_valid[window.indices()]
        subset_label = label_valid[window.indices()]
        hsi_test_list.append(subset_hsi)
        msi_test_list.append(subset_msi)
        sar_test_list.append(subset_sar)
        label_test_list.append(subset_label)
    del hsi_valid, msi_valid, sar_valid, label_valid
    hsi_test_list = np.array(hsi_test_list)
    msi_test_list = np.array(msi_test_list)
    sar_test_list = np.array(sar_test_list)
    label_test_list = np.array(label_test_list)

    # construct dataset
    hsi_list = torch.from_numpy(hsi_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    msi_list = torch.from_numpy(msi_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    sar_list = torch.from_numpy(sar_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    label_list = torch.from_numpy(label_list).type(torch.LongTensor)
    label_train = Data.TensorDataset(hsi_list, msi_list, sar_list, label_list)

    hsi_valid_list = torch.from_numpy(hsi_valid_list[:, :, :, :band].transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    msi_valid_list = torch.from_numpy(msi_valid_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    sar_valid_list = torch.from_numpy(sar_valid_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    label_valid_list = torch.from_numpy(label_valid_list).type(torch.LongTensor)
    label_valid = Data.TensorDataset(hsi_valid_list, msi_valid_list, sar_valid_list, label_valid_list)

    hsi_test_list = torch.from_numpy(hsi_test_list[:, :, :, :band].transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    msi_test_list = torch.from_numpy(msi_test_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    sar_test_list = torch.from_numpy(sar_test_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    label_test_list = torch.from_numpy(label_test_list).type(torch.LongTensor)
    label_test = Data.TensorDataset(hsi_test_list, msi_test_list, sar_test_list, label_test_list)

    return label_train, label_valid, label_test, num_classes, band


def slide_crop_all_modalities(dataset, patch, overlay):
    hsi, msi, sar, label, hsi_valid, msi_valid, sar_valid, label_valid, num_classes, band = read_data(dataset)

    if dataset == 'augsburg':
        col_train, row_train = 1360, 886
        col_valid, row_valid = 811, 2465

    elif dataset == 'beijing':
        col_train, row_train = 8706, 13474
        col_valid, row_valid = 8670, 6225

    else:
        raise ValueError("Unknown dataset")

    # slide crop for train data
    img = np.concatenate([hsi, msi, sar], axis=2)  # w, h, c
    del hsi, msi, sar
    window_set_train = sw.generate(img, sw.DimOrder.HeightWidthChannel, patch, overlay)
    img_list = []
    label_list = []
    for window in window_set_train:
        subset_img = img[window.indices()]
        subset_label = label[window.indices()]
        img_list.append(subset_img)
        label_list.append(subset_label)
    img_list = np.array(img_list)
    label_list = np.array(label_list)
    del img, label

    # non-overlay crop for valid data
    img_valid = np.concatenate([hsi_valid, msi_valid, sar_valid], axis=2)
    del hsi_valid, msi_valid, sar_valid

    window_set_valid = sw.generate(img_valid, sw.DimOrder.HeightWidthChannel, patch, 0)
    img_valid_list = []
    label_valid_list = []
    for window in window_set_valid:
        subset_img = img_valid[window.indices()]
        subset_label = label_valid[window.indices()]
        img_valid_list.append(subset_img)
        label_valid_list.append(subset_label)
    img_valid_list = np.array(img_valid_list)
    label_valid_list = np.array(label_valid_list)
    del img_valid, label_valid

    # construct dataset
    img_list = torch.from_numpy(img_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    label_list = torch.from_numpy(label_list).type(torch.LongTensor)
    label_train = Data.TensorDataset(img_list, label_list)

    img_valid_list = torch.from_numpy(img_valid_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    label_valid_list = torch.from_numpy(label_valid_list).type(torch.LongTensor)
    label_valid = Data.TensorDataset(img_valid_list, label_valid_list)

    return label_train, label_valid, num_classes, band
