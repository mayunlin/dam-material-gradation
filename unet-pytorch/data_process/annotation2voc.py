import os
import shutil


def makefolder(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    train_base = '/home/chase/Boyka/data/stone1117/VOC'
    VOCdevkit_path = '/home/chase/Boyka/data/stone1117/ours/VOC'
    print("Generate txt in ImageSets.")
    saveImgSetPath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    tarImgPath = os.path.join(VOCdevkit_path, 'VOC2007/JPEGImages')
    tarMaskPath = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    makefolder(saveImgSetPath)
    makefolder(tarImgPath)
    makefolder(tarMaskPath)
    ftrainval = open(os.path.join(saveImgSetPath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveImgSetPath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveImgSetPath, 'train.txt'), 'w')
    fval = open(os.path.join(saveImgSetPath, 'val.txt'), 'w')

    train_seg_path = os.path.join(train_base, 'SegmentationClass')
    train_img_path = os.path.join(train_base, 'JPEGImages')
    for seg_name in os.listdir(train_seg_path):
        save_name = seg_name.split('.')[0]
        src_mask = os.path.join(train_seg_path, seg_name)
        tar_mask = os.path.join(tarMaskPath, save_name + '.png')
        shutil.copyfile(src_mask, tar_mask)
        src_img = os.path.join(train_img_path, seg_name.replace('.png', '.jpg'))
        tar_img = os.path.join(tarImgPath, save_name + '.jpg')
        shutil.copyfile(src_img, tar_img)
        ftrainval.write(save_name + '\n')
        ftrain.write(save_name + '\n')

    test_seg_path = os.path.join(train_base, 'SegmentationClass')
    test_img_path = os.path.join(train_base, 'JPEGImages')
    for seg_name in os.listdir(test_seg_path):
        save_name = seg_name.split('.')[0]
        src_mask = os.path.join(test_seg_path, seg_name)
        tar_mask = os.path.join(tarMaskPath, save_name + '.png')
        shutil.copyfile(src_mask, tar_mask)
        src_img = os.path.join(test_img_path, seg_name.replace('.png', '.jpg'))
        tar_img = os.path.join(tarImgPath, save_name + '.jpg')
        shutil.copyfile(src_img, tar_img)
        fval.write(save_name + '\n')
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")
