import argparse
import os
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(
        description='Check relationship between Imagenet-21k and Imagenet-1k')
    parser.add_argument('src', help='Imagenet-1k path')
    parser.add_argument('dst', help='Imagenet-21k path')
    args = parser.parse_args()
    return args

"""
├── src
│   ├── train
│   │   ├── n01440764
│   │   │   ├── n01440764_10026.JPEG
│   │   │   ├── n01440764_10027.JPEG
|   |   |   ├── n01440764_10028.JPEG
│   ├── val
│   │   ├── n01440764
│   │   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   │   ├── ILSVRC2012_val_00002138.JPEG
|   |   |   ├── ILSVRC2012_val_00003014.JPEG

├── dst
│   ├── n00004475
│   │   ├── n00004475_15899.JPEG
│   │   ├── n00004475_32312.JPEG
│   │   ├── n00004475_35466.JPEG
"""

def main():
    args = parse_args()
    dst_files = set()
    dst_classes = set()
    for root, dirs, files in os.walk(args.dst):
        dst_classes.update(dirs)
        dst_files.update(files)
    src_train_files = set()
    src_train_classes = set()
    for root, dirs, files in os.walk(osp.join(args.src, 'train')):
        src_train_classes.update(dirs)
        src_train_files.update(files)
    src_val_files = set()
    src_val_classes = set()
    for root, dirs, files in os.walk(osp.join(args.src, 'val')):
        src_val_classes.update(dirs)
        src_val_files.update(files)

    print('Statistics about ImageNet-21k')
    print(f'Number of classes: {len(dst_classes)}')
    print(f'Number of images: {len(dst_files)}')

    print('\n')
    print('Statistics about ImageNet-1k')
    print(f'Number of training classes: {len(src_train_classes)}')
    print(f'Number of training images: {len(src_train_files)}')
    print(f'Number of validation classes: {len(src_val_classes)}')
    print(f'Number of validation images: {len(src_val_files)}')
    print(f'Overlap between training and validation classes: '
          f'{len(src_train_classes.intersection(src_val_classes))}')
    print(f'Overlap between training and validation images: '
          f'{len(src_train_files.intersection(src_val_files))}')

    print('\n')
    print('Difference between ImageNet-21k and ImageNet')
    print(f'Class overlap with ImageNet-1k train: '
          f'{len(dst_classes.intersection(src_train_classes))}')
    print(f'Unique class in ImageNet-21k compared with ImageNet-1k train: '
          f'{len(dst_classes - src_train_classes)}')
    print(f'Unique class in ImageNet-1k train compared with ImageNet-21k: '
          f'{len(src_train_classes - dst_classes)}, {src_train_classes - dst_classes}')
    print(f'Class overlap with ImageNet-1k val: '
          f'{len(dst_classes.intersection(src_val_classes))}')
    print(f'Unique class in ImageNet-21k compared with ImageNet-1k val: '
          f'{len(dst_classes - src_val_classes)}')
    print(f'Unique class in ImageNet-1k val compared with ImageNet-21k: '
          f'{len(src_val_classes - dst_classes)}')

    print('\n')
    print(f'File overlap with ImageNet-1k train: '
          f'{len(dst_files.intersection(src_train_files))}')
    print(f'Unique files in ImageNet-21k compared with ImageNet-1k train: '
          f'{len(dst_files - src_train_files)}')
    print(f'Unique files in ImageNet-1k train compared with ImageNet-21k: '
          f'{len(src_train_files - dst_files)}')
    print(f'File overlap with ImageNet-1k val: '
          f'{len(dst_files.intersection(src_val_files))}')
    print(f'Unique file in ImageNet-21k compared with ImageNet-1k val: '
          f'{len(dst_files - src_val_files)}')
    print(f'Unique file in ImageNet-1k val compared with ImageNet-21k: '
          f'{len(src_val_files - dst_files)}')




if __name__ == '__main__':
    main()
