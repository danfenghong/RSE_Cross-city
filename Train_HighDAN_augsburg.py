import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import datetime
from networks.HighDAN import HighResolutionNet
from networks.discriminator import FCDiscriminator, OutspaceDiscriminator
from utils.highDHA_utils import adjust_learning_rate, adjust_learning_rate_D
from utils.pyt_utils import compute_cm, compute_IoU, compute_mIoU, compute_OA, compute_f1, compute_kappa, plot, recover
from loss.criterion import CrossEntropy, DiceLoss
import numpy as np
import pandas as pd
import argparse
from data import dataset
import random
import os
import csv
import matplotlib.pyplot as plt
import shutil

parser = argparse.ArgumentParser(description='Cross-City Semantic Segmentation')
parser.add_argument('--fix_random', default=True, help='fix randomness')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--gpu_id', default='6', help='gpu id')
# a 6000 beijing 1000
parser.add_argument('--epoch', default=6000, type=int, help='number of epoch')
parser.add_argument('--loop_epoch', default=1, type=int, help='number of epochs per validate')

parser.add_argument('--model', default='HighDAN', choices=['HighDAN'], type=str)

# dataset parameters
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--patch', default=128, type=int, help='input data size')
parser.add_argument('--overlay', default=0.5, type=float, help='overlay size')
parser.add_argument('--dataset', choices=['augsburg', 'beijing'], default='augsburg', type=str, help='dataset to use')
parser.add_argument('--pca_flag', default=True, type=bool, help='weather use PCA dimension reduction on dataset')  # 10
parser.add_argument('--band_norm_flag', default=True, help='normalization by band')
parser.add_argument('--backbone_flag', default=True, help='use_backbone_pretrain')
parser.add_argument('--aug_flag', default=True, help='use_augment')
parser.add_argument('--backbone_path', default='./pretrain/hrnetv2_w48_imagenet_pretrained.pth', help='use_backbone')
# loss
parser.add_argument('--add_dice', default=True, type=bool, help='loss func')
# optimizer parameters
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--learning_rate_D', default=1e-4, type=float)
parser.add_argument('--lr_decay', default=20, type=int)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--power', default=0.9, type=float)
args = parser.parse_args()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    a = torch.cuda.is_available()
    if torch.cuda.is_available():
        print('GPU is true')
        print('Cuda Version: {}'.format(torch.version.cuda))
    else:
        print('CPU is true')

    if args.fix_random:
        manualSeed = args.seed
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

        cudnn.deterministic = True
        cudnn.benchmark = False

    else:
        cudnn.benchmark = True
    print('Using GPU: {}'.format(args.gpu_id))

    # create dataset and model
    results_file = "results_{}.txt".format(args.model+args.dataset)
    label_train_loader, label_valid_loader, label_test_loader, num_classes, band = dataset.getdata(
        args.dataset,
        args.patch,
        args.overlay,
        args.batch_size,
        args.pca_flag,
        args.band_norm_flag,
        args.aug_flag
    )

    criterion_ce = CrossEntropy(ignore_label=0)
    # denominator is the square term
    criterion_dice = DiceLoss(ignore_label=0, smooth=1e-5, p=1)
    criterion_mse = nn.MSELoss()
    # create model
    model = HighResolutionNet(band, num_classes).cuda()
    # load pretrained backbone
    if args.backbone_flag:
        saved_state_dict = torch.load(args.backbone_path)
        model_dict = model.state_dict()
        for k, v in list(saved_state_dict.items()):
            if k == 'conv1.weight':
                saved_state_dict.pop(k)
            if k == 'conv1.bias':
                saved_state_dict.pop(k)
            if k == 'bn1.weight':
                saved_state_dict.pop(k)
            if k == 'bn1.bias':
                saved_state_dict.pop(k)
            if k == 'bn1.running_mean':
                saved_state_dict.pop(k)
            if k == 'bn1.running_var':
                saved_state_dict.pop(k)
            if k == 'bn1.num_batches_tracked':
                saved_state_dict.pop(k)
            if k == 'conv2.weight':
                saved_state_dict.pop(k)
            if k == 'conv2.bias':
                saved_state_dict.pop(k)
            if k == 'bn2.weight':
                saved_state_dict.pop(k)
            if k == 'bn2.bias':
                saved_state_dict.pop(k)
            if k == 'bn2.running_mean':
                saved_state_dict.pop(k)
            if k == 'bn2.running_var':
                saved_state_dict.pop(k)
            if k == 'bn2.num_batches_tracked':
                saved_state_dict.pop(k)
            if str.find(k, 'last_layer') != -1:
                saved_state_dict.pop(k)
        saved_state_dict = {k: v for k, v in saved_state_dict.items()
                            if k in model_dict.keys()}
        model_dict.update(saved_state_dict)
        misskey, _ = model.load_state_dict(model_dict, strict=False)
        layer1 = model.layer1.state_dict()
        model.msi_layer1.load_state_dict(layer1)
        model.sar_layer1.load_state_dict(layer1)
        print("load pretrained backbone")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # discriminator attention module
    num_class_list = [720*3, num_classes]
    model_DA = nn.ModuleList(
        [FCDiscriminator(num_classes=num_class_list[0]).cuda(),
         OutspaceDiscriminator(num_classes=num_class_list[1]).cuda()]
    )
    optimizer_D = torch.optim.Adam(model_DA.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))

    # label for adversarial training
    source_label = 0
    target_label = 1

    save_root_path = os.path.join(os.path.abspath(''),
                                  'result', args.model,
                                  args.dataset, "add_dice" if args.add_dice else "no_dice",
                                  "add_pca" if args.pca_flag else "no_pca",
                                  "band_norm" if args.band_norm_flag else "all_norm",
                                  "bz_" + str(args.batch_size),
                                  "use_pretrain" if args.backbone_flag else "no_pretrain",
                                  "aug" if args.aug_flag else "no_aug"
                                  )

    if os.path.exists(save_root_path):
        shutil.rmtree(save_root_path)
    miou_score = []
    max_miou = 0

    for epoch in range(args.epoch):
        model.train()
        model_DA[0].train()
        model_DA[1].train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, args.learning_rate, epoch, args.epoch, args.power)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, args.learning_rate_D, epoch, args.epoch, args.power)

        # train G
        # don't accumulate grads in D
        for param in model_DA.parameters():
            param.requires_grad = False

        # train with source
        try:
            _, traindata = next(enumerate(label_train_loader))
        except StopIteration:
            trainloader_iter = iter(label_train_loader)
            _, traindata = next(trainloader_iter)

        x, y, z, label = traindata
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()
        label = label.cuda()

        feat_source, pred_source = model(x, y, z, model_DA, 'source')
        loss = criterion_ce(pred_source, label)
        if args.add_dice:
            loss += criterion_dice(pred_source, label)
        loss.backward()

        # train with target
        try:
            _, validdata = next(enumerate(label_valid_loader))
        except StopIteration:
            targetloader_iter = iter(label_valid_loader)
            _, validdata = next(targetloader_iter)

        x, y, z, label_val = validdata
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()
        feat_target, pred_target = model(x, y, z, model_DA, 'target')

        loss_adv = 0
        D_out = model_DA[0](feat_target)
        loss_adv += criterion_mse(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())
        D_out = model_DA[1](F.softmax(pred_target, dim=1))
        loss_adv += criterion_mse(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())
        loss_adv = loss_adv * 0.01
        loss_adv.backward()

        optimizer.step()

        # train D
        # bring back requires_grad
        for param in model_DA.parameters():
            param.requires_grad = True

        # train with source
        loss_D_source = 0
        D_out_source = model_DA[0](feat_source.detach())
        loss_D_source += criterion_mse(D_out_source,
                                       torch.FloatTensor(D_out_source.data.size()).fill_(source_label).cuda())
        D_out_source = model_DA[1](F.softmax(pred_source.detach(), dim=1))
        loss_D_source += criterion_mse(D_out_source,
                                       torch.FloatTensor(D_out_source.data.size()).fill_(source_label).cuda())
        loss_D_source.backward()

        # train with target
        loss_D_target = 0
        D_out_target = model_DA[0](feat_target.detach())
        loss_D_target += criterion_mse(D_out_target,
                                       torch.FloatTensor(D_out_target.data.size()).fill_(target_label).cuda())
        D_out_target = model_DA[1](F.softmax(pred_target.detach(), dim=1))
        loss_D_target += criterion_mse(D_out_target,
                                       torch.FloatTensor(D_out_target.data.size()).fill_(target_label).cuda())
        loss_D_target.backward()

        optimizer_D.step()

        print("Epoch: {:03d}, loss_seg: {:.4f}, loss_adv: {:.4f}, loss_D_s: {:.4f}, loss_D_t: {:.4f}"
              .format(epoch + 1, loss, loss_adv, loss_D_source, loss_D_target))

        if (epoch + 1) % args.loop_epoch == 0:
            model.eval()
            model_DA[0].eval()
            model_DA[1].eval()

            label_total = []
            label_gt = []
            for i, testdata in enumerate(label_test_loader):
                x, y, z, label = testdata

                x = x.cuda()
                y = y.cuda()
                z = z.cuda()
                label = label.cuda()
                _, output_label = model(x, y, z, model_DA, 'target')

                pred = output_label.cpu().detach().numpy().transpose(0, 2, 3, 1)
                seg_pred = np.asarray(np.argmax(pred[:, :, :, 1:], axis=3), dtype=np.uint8)
                label_total.append(seg_pred + 1)

                label = label.cpu().detach().numpy()
                label_gt.append(label)
                # print("valid epoch: {:03d}".format(i))

            save_path = os.path.join(save_root_path, "epoch_" + str(epoch + 1))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pd.DataFrame(np.array(label_gt).flatten()).to_csv(save_path + '/label_gt.csv', index=False)
            pd.DataFrame(np.array(label_total).flatten()).to_csv(save_path + '/label_pre.csv', index=False)
            label_gt, label_total = recover(label_gt, label_total, args.dataset, args.patch)
            cm = compute_cm(label_gt, label_total)
            pd.DataFrame(cm).to_csv(save_path + '/confuse_matrix.csv', index=True)

            IoU = compute_IoU(cm)
            mIoU = compute_mIoU(cm)
            f1 = compute_f1(label_total, label_gt)
            oa = compute_OA(cm)
            kappa = compute_kappa(label_total, label_gt)
            IoU_str = [f'{item:.4f}' for item in IoU]

            if mIoU > max_miou:
                max_miou = mIoU
                max_miou_epoch = epoch
                plot(label_gt, label_total, save_path)
                print("Epoch: {:03d}, max_miou_Epoch: {:03d}, max_miou: {:.4f}".format(epoch + 1, max_miou_epoch + 1, max_miou))
                print("save best weight:{:03d}".format(epoch + 1))
                torch.save(model.state_dict(), os.path.join(save_root_path, 'best_model.pth'))
                torch.save(model_DA[0].state_dict(), os.path.join(save_root_path, 'best_DA_1.pth'))
                torch.save(model_DA[1].state_dict(), os.path.join(save_root_path, 'best_DA_2.pth'))
                with open(results_file, "a") as f:
                    # 记录每个epoch对应的train_loss、lr以及验证集各指标
                    info = f"[epoch: {max_miou_epoch + 1}]\n" \
                           f"max_miou: {mIoU:.6f}\n"
                    f.write(info + "\n\n")
            with open(save_path + "/eval_index.csv", 'w', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["IoU:"])
                csv_writer.writerow([IoU_str])
                csv_writer.writerow(["mIoU: {:.4f}".format(mIoU)])
                csv_writer.writerow(["f1: {:.4f}".format(f1)])
                csv_writer.writerow(["OA: {:.4f}".format(oa)])
                csv_writer.writerow(["kappa: {:.4f}".format(kappa)])
            miou_score.append(mIoU)

    pd.DataFrame(miou_score).to_csv(os.path.join(save_root_path, 'miou_score.csv'), index=False, header=None)

    x = list(np.arange(0, args.epoch, args.loop_epoch) + args.loop_epoch)
    fig = plt.figure()
    plt.plot(x, miou_score)
    fig.savefig(os.path.join(save_root_path, 'miou_score.jpg'), dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    main()
