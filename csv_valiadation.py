import argparse
import torch
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval
assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))
#./csv_retinanet_undis_19_0.2291315644979477.pt
def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--csv_val', default="F:/VOC_Annotations_val_1.csv", help='Path to COCO directory')
    parser.add_argument('--model_path', default='./csv_retinanet_undis_19_0.2291315644979477.pt', help='Path to model', type=str)
    parser.add_argument('--csv_classes', default="F:/VOC_label.csv")
    parser = parser.parse_args(args)

    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                             transform=transforms.Compose([Normalizer(), Resizer()]))
    # Create the model
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    use_gpu = True
    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()
    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
        retinanet.load_state_dict(torch.load(parser.model_path))
        #retinanet.module = torch.load(parser.model_path)
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)
    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()
    csv_eval.evaluate(dataset_val, retinanet)
if __name__ == '__main__':
    main()
