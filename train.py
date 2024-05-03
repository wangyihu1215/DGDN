import argparse
import os

import torch
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

import dataset_transform
from nets import Net
from loss import CharbonnierLoss, Laplacian_Loss, VGG_loss

from dataset_loader import ImageFolder
from utils import AvgLoss

cudnn.benchmark = True


def train(config):
    # datasets load
    data_transform = dataset_transform.Compose([
        dataset_transform.Resize(config.img_size),
        dataset_transform.RandomHorizontallyFlip(),
        dataset_transform.ToTensor()
    ])

    train_set = ImageFolder(config.root, data_transform=data_transform, is_train=True)
    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              num_workers=config.num_workers, shuffle=True)

    # net load
    net = Net().cuda().train()

    # loss load
    Char_loss = CharbonnierLoss()
    Lapl_loss = Laplacian_Loss()
    Vgg_loss = VGG_loss()

    # optimizer load
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * config.lr},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': config.lr, 'weight_decay': config.weight_decay}
    ])

    # checkpoint folder
    if not os.path.exists(config.checkpoint):
        os.mkdir(config.checkpoint)
    log_path = os.path.join(config.checkpoint, config.log_name)

    # train
    curr_iter = 0

    while True:
        train_loss_record = AvgLoss()
        train_loss_dehaze_record = AvgLoss()
        train_loss_depth_record = AvgLoss()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * config.lr * (1 - float(curr_iter) / config.iter_num
                                                               ) ** config.lr_decay
            optimizer.param_groups[1]['lr'] = config.lr * (1 - float(curr_iter) / config.iter_num
                                                           ) ** config.lr_decay

            original, haze, depth = data

            batch_size = config.batch_size

            original = original.cuda()
            haze = haze.cuda()
            depth = depth.cuda()

            optimizer.zero_grad()

            result, depth_pred = net(haze)

            # loss dehaze
            loss_char_dehaze = Char_loss(result, original)

            loss_lapl_dehaze = Lapl_loss(result, original)
            loss_vgg_dehaze = Vgg_loss(result, original)

            loss_dehaze = loss_char_dehaze + 0.8 * (loss_lapl_dehaze + loss_vgg_dehaze)

            # loss depth
            loss_char_depth = Char_loss(depth_pred, depth)

            depth_pred = depth_pred.repeat(1, 3, 1, 1)
            depth = depth.repeat(1, 3, 1, 1)
            loss_lapl_depth = Lapl_loss(depth_pred, depth)
            loss_vgg_depth = Vgg_loss(depth_pred, depth)

            loss_depth = loss_char_depth + 0.8 * (loss_lapl_depth + loss_vgg_depth)

            # loss total
            loss = loss_dehaze + loss_depth

            loss.backward()

            optimizer.step()

            train_loss_record.update(loss.data, batch_size)
            train_loss_dehaze_record.update(loss_dehaze.data, batch_size)
            train_loss_depth_record.update(loss_depth.data, batch_size)

            curr_iter += 1

            log = '[iter %d], [train loss %.9f], [lr %.13f], [loss_dehaze %.9f], [loss_depth %.9f]' % \
                  (curr_iter, train_loss_record.avg, optimizer.param_groups[1]['lr'],
                   train_loss_dehaze_record.avg, train_loss_depth_record.avg)
            print(log)
            open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % config.snapshot_epoch == 0:
                torch.save(net.state_dict(), os.path.join(config.checkpoint, ('%d.pth' % (curr_iter + 1))))
                torch.save(optimizer.state_dict(),
                           os.path.join(config.checkpoint, ('%d_optim.pth' % (curr_iter + 1))))

            if curr_iter > config.iter_num:
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default='F:/Datasets/Train/')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--log_name', type=str, default='depth_record.txt')
    parser.add_argument('--iter_num', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--img_size', type=list, default=[1024, 512], help='resized img size [w, h]')
    parser.add_argument('--snapshot_epoch', type=int, default=10000)

    config = parser.parse_args()

    train(config)
