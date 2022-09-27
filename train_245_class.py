import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# todo： Change to your dataset path
def data_load(data_dir="D:/code/trash_jpg_renamed_new"):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),#是把图像按照中心随机切割成224正方形大小的图片。
            transforms.RandomHorizontalFlip(),#水平翻转
            transforms.ToTensor(),# 转换为tensor格式，这个格式可以直接输入进神经网络了。
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#对像素值进行归一化处理。
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),#按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。
            transforms.CenterCrop(224),#中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # data_dir = 'data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=42, shuffle=True, num_workers=0)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    # print(dataloaders['train'].batch_size)
    return dataloaders, dataset_sizes, class_names


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=35):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        # 每个 epoch 都有训练和验证的过程
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练阶段
            else:
                model.eval()   # 验证阶段

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            with tqdm(total=dataset_sizes[phase]) as pbar:
                for inputs, labels in dataloaders[phase]:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 使参数梯度归零
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # 只在训练阶段进行后向传播 + 优化
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 统计
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    # 更新进度条
                    pbar.update(dataloaders[phase].batch_size)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        print()
    '''
    plotCurve(range(1, num_epochs + 1), train_ls,
              "epoch", "loss",
              range(1, num_epochs + 1), test_ls,
              ["train", "test"])
    '''


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


def train_main():
    dataloaders, dataset_sizes, class_names = data_load()
    print(class_names)
    # 使用mobilenet模型
    model = models.mobilenet_v2(pretrained=True)

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)
    print(model)
    # num_ftrs = model_ft.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model_ft.fc = nn.Linear(num_ftrs, 2)
    #
    # model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 每7个 epoch 衰减lr * 0.1倍
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_bset = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
    torch.save(model_bset, "./models/my_train_mobilenet_trashv1_2_50.pt")
    torch.save(model_bset, "./models/my_train_mobilenet_trashv1_2_50.pth")
    torch.save(model_bset, "./models/my_train_mobilenet_trashv1_2_50.pkl")


if __name__ == '__main__':
    train_main()
