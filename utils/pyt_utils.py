# encoding: utf-8

import time
from collections import OrderedDict, defaultdict
import torch
from .logger import get_logger
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn import metrics
from skimage import color
import matplotlib
import matplotlib.pyplot as plt
from torch.nn import init
import math
from PIL import Image

logger = get_logger()


def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        device = torch.device('cpu')
        state_dict = torch.load(model_file, map_location=device)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model


def compute_cm(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask = (y_true > 0) & (y_true <= 13)
    cm = metrics.confusion_matrix(y_true[mask], y_pred[mask])
    return cm


def compute_IoU(confusion_matrix):
    intersection = np.diag(confusion_matrix)  # 交集
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)  # 并集
    IoU = intersection / union  # 交并比，即IoU
    return IoU


def compute_mIoU(confusion_matrix):
    intersection = np.diag(confusion_matrix)  # 交集
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)  # 并集
    IoU = intersection / union  # 交并比，即IoU
    MIoU = np.mean(IoU)  # 计算MIoU
    return MIoU


def compute_f1(prediction, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    # prediction.tolist(), target.tolist()

    img, target = np.array(prediction).flatten(), np.array(target).flatten()
    mask = (target > 0) & (target <= 13)
    f1 = metrics.f1_score(y_true=target[mask], y_pred=img[mask], average='macro')
    return f1


def compute_f1_1(confusion_matrix):
    precision = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    recall = np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)
    f1score = 2 * precision * recall / (precision + recall)
    # f1score = torch.where(torch.isnan(f1score), torch.full_like(f1score, 0), f1score)
    f1score = np.nan_to_num(f1score)
    mf1score = np.mean(f1score)
    return mf1score, f1score


def compute_kappa(prediction, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        kappa: float
    """
    # prediction.tolist(), target.tolist()
    img, target = np.array(prediction).flatten(), np.array(target).flatten()
    mask = (target > 0) & (target <= 13)
    kappa = metrics.cohen_kappa_score(target[mask], img[mask])
    return kappa


def compute_OA(confusion_matrix):
    OA = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return OA


def patch_concat(patch_label, h, w, patch_size=128):
    patch_label = torch.tensor(patch_label)
    row, col = patch_label.size(0), patch_label.size(1)
    patch_list = []
    for i in range(row):  # 0-7
        patch = patch_label[i, 0, :, :]
        for j in range(1, col):  # 1-19
            cur_patch = patch_label[i, j, :, :]
            if j == col - 1:
                cur_patch = cur_patch[(patch_size * j - h):, :]
            patch = torch.concat([patch, cur_patch], dim=0)
        if i == row - 1:
            patch = patch[:, (patch_size * i - w):]
        patch_list.append(patch)
    return torch.concat(patch_list, dim=1)


def recover(y_true, y_pred, dataset, patch_size):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask = (y_true > 0) & (y_true <= 13)
    # mask1 = mask.astype(np.uint8)
    # mask2 = Image.fromarray(mask1)
    # mask2.save("./mask_gt.png")
    y_pred = mask * y_pred
    if dataset == 'augsburg':
        h, w = 2456, 811
    elif dataset == 'beijing':
        h, w = 6225, 8670
    else:
        raise ValueError("reset dataset")
    row, col = math.ceil(h / patch_size), math.ceil(w / patch_size)
    y_true = y_true.reshape((col, row, patch_size, patch_size))
    y_pred = y_pred.reshape((col, row, patch_size, patch_size))
    y_true = patch_concat(y_true, h, w, patch_size)
    y_pred = patch_concat(y_pred, h, w, patch_size)

    return y_true, y_pred

def recover1(y_true, y_pred, dataset, patch_size):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask = (y_true > 0) & (y_true <= 13)
    # mask1 = mask.astype(np.uint8)
    # mask2 = Image.fromarray(mask1)
    # mask2.save("./mask_gt.png")
    y_pred = mask * y_pred
    if dataset == 'augsburg':
        h, w = 886, 1360
    elif dataset == 'beijing':
        h, w = 6225, 8670
    else:
        raise ValueError("reset dataset")
    row, col = math.ceil(h / patch_size), math.ceil(w / patch_size)
    y_true = y_true.reshape((col, row, patch_size, patch_size))
    y_pred = y_pred.reshape((col, row, patch_size, patch_size))
    y_true = patch_concat(y_true, h, w, patch_size)
    y_pred = patch_concat(y_pred, h, w, patch_size)

    return y_true, y_pred

def plot(y_true, y_pred, path):
    colors = ('cyan', 'white', 'red', 'plum', 'darkviolet', 'magenta', 'yellow',
              'peru', 'darkkhaki', 'lime', 'yellowgreen', 'saddlebrown', 'darkslateblue')
    color_dic = {0: "#000000",
                 1: '#00FFFF',
                 2: '#FFFFFF',
                 3: '#FF0000',
                 4: '#DDA0DD',
                 5: '#9400D3',
                 6: '#FF00FF',
                 7: '#FFFF00',
                 8: '#CD853F',
                 9: '#BDB76B',
                 10: '#00FF00',
                 11: '#9ACD32',
                 12: '#8B4513',
                 13: '#483D8B'}

    label_dic = {0: 'Background',
                 1: 'Surface water',
                 2: 'Street',
                 3: 'Urban Fabric',
                 4: 'Industrial, commercial and transport',
                 5: 'Mine, dump and construction sites',
                 6: 'Artificial, vegetated areas',
                 7: 'Arable Land',
                 8: 'Permanent Crops',
                 9: 'Pastures',
                 10: 'Forests',
                 11: 'Shrub',
                 12: 'Open spaces with no vegetation',
                 13: 'Inland wetlands'}

    fig = plt.figure()
    # 图例
    legend_handles = [
        matplotlib.lines.Line2D(
            [],
            [],
            marker="s",
            color="w",
            markerfacecolor=color_dic[yi],
            ms=10,
            alpha=1,
            linewidth=0,
            label=label_dic[yi],
            markeredgecolor="k",
        )
        for yi in label_dic.keys()
    ]
    legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    plt.legend(handles=legend_handles, **legend_kwargs_)

    # image 1
    plt.subplot(1, 2, 1)
    dst_train = color.label2rgb(np.array(y_true), colors=colors, bg_label=0)
    plt.title('ground truth')
    plt.imshow(dst_train)
    # io.imsave(os.path.join('./data', 'train_label_map.jpg'), dst_train)
    # io.show()

    plt.subplot(1, 2, 2)
    dst_test = color.label2rgb(np.array(y_pred), colors=colors, bg_label=0)
    plt.title('predict')
    plt.imshow(dst_test)
    # io.imsave(os.path.join('./data', 'test_label_map.jpg'), dst_test)
    # io.show()
    fig.savefig(os.path.join(path, "label_map_recover.jpg"), dpi=600, bbox_inches='tight')
    plt.close(3)


def compute_hist(pred, lb, n_classes):
    ignore_label = 0
    keep = np.logical_not(lb == ignore_label)
    merge = pred[keep] * n_classes + lb[keep]
    hist = np.bincount(merge, minlength=n_classes ** 2)
    hist = hist.reshape((n_classes, n_classes))
    return hist


def computer_eval_index(preds, gts, n_classes=13):
    hist_size = (n_classes, n_classes)
    hist = np.zeros(hist_size, dtype=np.float32)

    for pred, gt in preds, gts:
        hist_once = compute_hist(pred, gt)
        hist += hist_once

    Precision = np.diag(hist) / np.sum(hist, axis=1)
    Recall = np.diag(hist) / np.sum(hist, axis=0)
    F1_score = 2 * Precision * Recall / (Precision + Recall)
    IoUs = np.diag(hist) / (np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist))
    mIoU = np.mean(IoUs)
    OA = np.diag(hist).sum() / hist.sum()
    IoU_str = [f'{item:.4f}' for item in IoUs]
    print("IoU:\n", "\t".join(IoU_str))
    print("mIoU: {:.4f}".format(mIoU))
    print("f1: {:.4f}".format(F1_score))
    print("OA: {:.4f}".format(OA))
