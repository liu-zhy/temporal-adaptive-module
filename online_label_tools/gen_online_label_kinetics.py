# Code for Online Loading Kinetics-400 Dataset Label Generation
# This code is pulled by [HustQBW](https://github.com/HustQBW/)

import os
import cv2

VIDEO_PATH = '/yrfs2/ssd/public_data/data/public_data/kinetics/k400/'
label_path = '/home/intern/bwqu2/kinetics_labels/labels'

def get_video2path_dict(tag='train'):
    video2path_dict = dict()
    for p in os.listdir(VIDEO_PATH + tag):
        for v in os.listdir(os.path.join(VIDEO_PATH + tag,p)):
            video2path_dict[v[:-4]] = f'{VIDEO_PATH + tag}/{p}/{v}' # :-4代表去掉.mp4

    return video2path_dict

if __name__ == '__main__':
    with open('kinetics_label_map.txt') as f:
        categories = f.readlines()
        categories = [c.strip().replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '') for c in categories]
    assert len(set(categories)) == 400
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    print(dict_categories)

    files_input = ['val.csv', 'train.csv']
    files_output = ['k400_val_list.txt', 'k400_train_list.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        tag = filename_input[:-4] # 去掉.csv
        video2path_dict = get_video2path_dict(tag=tag)
        count_cat = {k: 0 for k in dict_categories.keys()}
        with open(os.path.join(label_path, filename_input)) as f:
            lines = f.readlines()[1:]
        folders = []
        idx_categories = []
        categories_list = []
        for line in lines:
            line = line.rstrip()
            items = line.split(',')
            # folders.append(items[1] + '_' + items[2])
            video_name = '{0}_{1:0>6d}_{2:0>6d}'.format(items[1],int(items[2]),int(items[3]))
            folders.append(video_name)

            this_catergory = items[0].replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '')
            categories_list.append(this_catergory)
            idx_categories.append(dict_categories[this_catergory])
            count_cat[this_catergory] += 1
        print(max(count_cat.values()))

        assert len(idx_categories) == len(folders)
        missing_folders = []
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            curPath = video2path_dict.get(curFolder)

            if curPath == None:
                missing_folders.append(curFolder+'_Not_Existing')
                # print(missing_folders)
            else:
                cap = cv2.VideoCapture(curPath)
                if cap.isOpened() != True:
                    missing_folders.append(curPath)
                    continue

                frames_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                cap.release()

                if frames_num == 0:
                    missing_folders.append(curPath)
                    continue

                output.append('%s %d %d' % (curPath, frames_num, curIDX))

            print('%d/%d, missing %d'%(i, len(folders), len(missing_folders)))
        with open(os.path.join(label_path, filename_output),'w') as f:
            f.write('\n'.join(output))
        with open(os.path.join(label_path, 'missing_' + filename_output),'w') as f:
            f.write('\n'.join(missing_folders))
