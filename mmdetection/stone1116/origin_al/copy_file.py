import os
import shutil
import json

if __name__ == '__main__':

    src_path = '/home/chase/Boyka/data/stone1117/origin_algorithms/train'
    tar_path = '/home/chase/Boyka/data/stone1117/origin_algorithms/train_10'
    for file_name in os.listdir(src_path):
        src = os.path.join(src_path, file_name)
        for idx in range(10):
            tar_file_name = file_name.split('.')[0] + '_' + str(idx) + '.' + file_name.split('.')[1]
            tar = os.path.join(tar_path, tar_file_name)
            if file_name.split('.')[1] == 'jpg':
                shutil.copyfile(src, tar)
            else:
                # info_dict = json.loads(src)
                with open(src, 'r', ) as fw:
                    info_dict = json.load(fw)
                info_dict['imagePath'] = tar_file_name.replace('.json', '.jpg')
                with open(tar, 'w', encoding='utf-8') as fw:
                    json.dump(info_dict, fw, indent=4, ensure_ascii=False)
