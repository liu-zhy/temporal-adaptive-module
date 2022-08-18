# Code for Online Loading Kinetics-400 Dataset Configuration
# This code is pulled by [HustQBW](https://github.com/HustQBW/)

import os

ROOT_DATASET = None


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        filename_imglist_train = '/home/intern/bwqu2/kinetics_labels/labels/k400_train_list.txt'
        filename_imglist_val = '/home/intern/bwqu2/kinetics_labels/labels/k400_val_list.txt'
    else:
        raise NotImplementedError('no such modality:' + modality)

    return filename_categories, filename_imglist_train, filename_imglist_val



def return_dataset(dataset, modality):
    dict_single = {
        'kinetics': return_kinetics
    }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val = dict_single[
            dataset](modality)
    else:
        raise ValueError('Unknown dataset ' + dataset)


    categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val
