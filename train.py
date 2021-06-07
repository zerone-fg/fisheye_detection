import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter_d, \
    Normalizer
from torch.utils.data import DataLoader
from retinanet import coco_eval
from retinanet import csv_eval
assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))
def valid():
    '''
  if parser.dataset == 'coco':

      print('Evaluating dataset')

      coco_eval.evaluate_coco(dataset_val, retinanet)

  elif parser.dataset == 'csv' and parser.csv_val is not None:

      print('Evaluating dataset')
      mAP = csv_eval.evaluate(dataset_val, retinanet)
      sum = 0
      count = 0
      for item in mAP.values():
          sum += item[0]
      mAP = sum / 20
  scheduler.step(np.mean(epoch_loss))
      '''
def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', default="csv", help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', default=None, help='Path to COCO directory')
    parser.add_argument('--csv_train', default="F:/VOC_Annotations_train_1.csv", help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', default="F:/VOC_label.csv", help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', default="F:/VOC_Annotations_val_1.csv", help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--checkpoint', default="csv_retinanet_undis_15_0.29325339034895315.pt", help='the model path saved')
    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter_d(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter_d(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=0, collate_fn=collater, batch_sampler=sampler)


    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True
    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()
    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else: 
        retinanet = torch.nn.DataParallel(retinanet)
    retinanet.training = True


    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()
    if parser.checkpoint:
        checkpoint = torch.load(parser.checkpoint)
        #retinanet.module = checkpoint
        print(retinanet.state_dict().keys())
        retinanet.load_state_dict(checkpoint, strict=False)
    print(retinanet.module)
    print('Num training images: {}'.format(len(dataset_train)))
    start_epoch = 16
    for epoch_num in range(start_epoch, parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['img_d'].cuda().float(), data['img_u'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue
            if iter_num % 6000 == 0:
                torch.save(retinanet.state_dict(), '{}_retinanet_undis_{}_{}.pt'.format(parser.dataset, epoch_num, np.mean(epoch_loss)))
    retinanet.eval()
    torch.save(retinanet, 'model_final.pt')

if __name__ == '__main__':
    main()
