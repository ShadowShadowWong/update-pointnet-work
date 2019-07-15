# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import sys
sys.path.append(r"/home/wxd/桌面/pointnet.pytorch")
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
# import sys
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

#这里是添加参数，即此程序可直接在命令行添加参数
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=50, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=25000, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

# opt.manualSeed = random.randint(1, 10000)  # fix seed使每次运行随机数是固定的
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)#这两个也是一样,应该是设置之后打乱顺序时
# torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=r'../trainset',
        classification=False,#change
        npoints=opt.num_points)
    # print("here-------------------\n")

    test_dataset = ShapeNetDataset(
        root=r'../testset',
        classification=False,
        # split='test',
        npoints=opt.num_points,
        data_augmentation=False)
    # print("here-------------------\n")

elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        drop_last=True,
        num_workers=int(opt.workers))

# print(len(dataset), len(test_dataset))
# num_classes = len(dataset.classes)#class的长度
# print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetDenseCls(k=8, feature_transform=opt.feature_transform)#77777
# classifier.load_state_dict(torch.load(r'cls/cls_model_2_300.pth'))#load the pram,跑了3编准确率可以高达90%，测试准确率达到80%
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))#加载模型

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)#每20步调整学习率，调整系数为0.5
# classifier.cuda()#转移到GPU进行计算

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):#这个只是重复跑几遍完整的数据集
    print("第%d遍：\n"%epoch)
    scheduler.step()
    for i, (points,target) in enumerate(dataloader):#这是一批32size的点云
        # print("len(target):", target.size())
        # print(target)
        lent = [len(target[0]), len(target)]
        # print(lent[0], lent[1])
        target = target.view(lent[0]*lent[1])
        # print("len(target):", target.size())

        # target = target[:, 0]
        points = points.transpose(2, 1)
        # points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()

        pred, trans, trans_feat = classifier(points)
        # print("len(pred):",pred.size())
        # print(pred)
        pred = pred.view(lent[0]*lent[1],8)
        # print("len(pred):",pred.size())
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize*25000)))
        if i % 5 == 0 and i!=0:
            j, (points, target) = next(enumerate(testdataloader, 0))
            # points, target = data
            lent = [len(target[0]), len(target)]
            # print(lent[0], lent[1])
            target = target.view(lent[0] * lent[1])
            # target = target[:, 0]
            points = points.transpose(2, 1)
            # points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            pred = pred.view(lent[0] * lent[1], 8)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
            epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(opt.batchSize * 25000)))
            torch.save(classifier.state_dict(), '%s/cls_model_%d_%d.pth' % (opt.outf, epoch, i))
            # print('save%s/cls_model_%d_%d.pth----------------------------------------' % (opt.outf, epoch, i))

    if(epoch!=0 and epoch%10==0):
        torch.save(classifier.state_dict(), '%s/cls_model_%d_%d.pth' % (opt.outf, epoch, i))
        print('save%s/cls_model_%d_%d.pth----------------------------------------' % (opt.outf, epoch, i))

        #每10批处理后进行测试
         #     if i % 100 == 0 and i!=0:
        #         torch.save(classifier.state_dict(), '%s/cls_model_%d_%d.pth' % (opt.outf, epoch, i))
total_correct = 0
total_testset = 0
#tqdm是进度条
#枚举，下标和value
#这里只是用来可视化
for i,(points, target) in tqdm(enumerate(testdataloader)):
    # points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    # points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))