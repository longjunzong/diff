import collections
import argparse
import torch
from mmcv.runner import CheckpointLoader


def filter_model(filename,map_location,logger=None):
    checkpoint=CheckpointLoader.load_checkpoint(filename, map_location, logger)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    state_dict1=collections.OrderedDict()
    for k, v in state_dict.items():
        if "student." in k:
            k=k[8:]
            state_dict1[k]=v

    checkpoint['state_dict']=state_dict1
    torch.save(checkpoint,"filtered_model/"+filename[-12:])

def print_paramters(filename,map_location,logger=None):
    checkpoint=CheckpointLoader.load_checkpoint(filename, map_location, logger)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    for k, v in state_dict.items():
        print(k)


def main():
    #parser = argparse.ArgumentParser(description='')
    #parser.add_argument('--show', type=int,default=0)
    #parser.add_argument('--epoch', type=int,default=12)
    #args = parser.parse_args()
    #filename="work_dirs/fgd_retina_r50_fpn_3x_distill_retina_r18_fpn_1x_coco/epoch_{}.pth".format(args.epoch)
    #show_name = "filtered_model/epoch_{}.pth".format(args.epoch)
    show_name = "/home/host/sy/ljz/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
    map_location="cpu"
    if True:
        print_paramters(show_name,map_location)
    else:
        filter_model(filename,map_location)

if __name__ == '__main__':
    main()
    #with open("999.txt","a") as f:
    #    f.write("123")
    #    f.write("\n")