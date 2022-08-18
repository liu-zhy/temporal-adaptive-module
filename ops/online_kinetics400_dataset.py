# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# Code for Online Loading Kinetics-400 Dataset
# This code is pulled by [HustQBW](https://github.com/HustQBW/)

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint
import cv2


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self,
                 root_path,
                 list_file,
                 num_segments=3,
                 new_length=1,
                 modality='RGB',
                 image_tmpl='img_{:05d}.jpg',
                 transform=None,
                 random_shift=True,
                 test_mode=False,
                 remove_missing=False,
                 dense_sample=False,
                 test_sample="dense-10"):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.test_sample = test_sample
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [
                    Image.open(
                        os.path.join(
                            self.root_path, directory,
                            self.image_tmpl.format(idx))).convert('RGB')
                ]
            except Exception:
                print(
                    'error loading image:',
                    os.path.join(self.root_path, directory,
                                 self.image_tmpl.format(idx)))
                return [
                    Image.open(
                        os.path.join(self.root_path, directory,
                                     self.image_tmpl.format(1))).convert('RGB')
                ]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(
                    os.path.join(self.root_path, directory,
                                 self.image_tmpl.format('x',
                                                        idx))).convert('L')
                y_img = Image.open(
                    os.path.join(self.root_path, directory,
                                 self.image_tmpl.format('y',
                                                        idx))).convert('L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(
                    os.path.join(
                        self.root_path, '{:06d}'.format(int(directory)),
                        self.image_tmpl.format(int(directory), 'x',
                                               idx))).convert('L')
                y_img = Image.open(
                    os.path.join(
                        self.root_path, '{:06d}'.format(int(directory)),
                        self.image_tmpl.format(int(directory), 'y',
                                               idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(
                        os.path.join(
                            self.root_path, directory,
                            self.image_tmpl.format(idx))).convert('RGB')
                except Exception:
                    print(
                        'error loading flow file:',
                        os.path.join(self.root_path, directory,
                                     self.image_tmpl.format(idx)))
                    flow = Image.open(
                        os.path.join(self.root_path, directory,
                                     self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]

        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            t_stride = 64 // self.num_segments
            # t_stride = 128 // self.num_segments
            sample_pos = max(
                1, 1 + record.num_frames - t_stride * self.num_segments)  # 300帧的话，值就是237
            # 相当于留下了末尾的64帧
            start_idx = 0 if sample_pos == 1 else np.random.randint(
                0, sample_pos - 1)  # 如果视频少于64帧，sample_pos就是1，则从头开始，高于64帧则随机选个地方开始
            # 如果视频高于64帧，预处理的时候会采样到64，这样的话sample_pos >=1 就随机从头部前面一些帧开始采样
            offsets = [(idx * t_stride + start_idx) % record.num_frames
                       for idx in range(self.num_segments)]
            # 少于64帧，offsets里会有重复（因为取模运算）
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length +
                                1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(
                    self.num_segments)), average_duration) + randint(
                    average_duration, size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(
                    randint(record.num_frames - self.new_length + 1,
                            size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            # t_stride = 8
            t_stride = 64 // self.num_segments
            # t_stride = 128 // self.num_segments
            sample_pos = max(
                1, 1 + record.num_frames - t_stride * self.num_segments)
            # start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx = sample_pos // 2
            offsets = [(idx * t_stride + start_idx) % record.num_frames
                       for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(
                    self.num_segments)
                offsets = np.array([
                    int(tick / 2.0 + tick * x)
                    for x in range(self.num_segments)
                ])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if "dense" in self.test_sample:
            num_clips = int(self.test_sample.split("-")[-1])
            # t_stride = 8
            t_stride = 64 // self.num_segments
            # t_stride = 128 // self.num_segments
            sample_pos = max(
                1, 1 + record.num_frames - t_stride * self.num_segments)
            if num_clips == 1:
                start_idx = sample_pos // 2
                offsets = [(idx * t_stride + start_idx) % record.num_frames
                           for idx in range(self.num_segments)]
            else:
                start_list = np.linspace(0,
                                         sample_pos - 1,
                                         num=num_clips,
                                         dtype=int)
                offsets = []
                for start_idx in start_list.tolist():
                    offsets += [
                        (idx * t_stride + start_idx) % record.num_frames
                        for idx in range(self.num_segments)
                    ]
            return np.array(offsets) + 1
        elif "uniform" in self.test_sample:
            num_clips = int(self.test_sample.split("-")[-1])
            if num_clips == 1:
                tick = (record.num_frames - self.new_length + 1) / float(
                    self.num_segments)
                offsets = [
                    int(tick / 2.0 + tick * x)
                    for x in range(self.num_segments)
                ]
            else:
                tick = (record.num_frames - self.new_length + 1) / float(
                    self.num_segments)
                start_list = np.linspace(0, tick - 1, num=num_clips, dtype=int)
                offsets = []
                # print(start_list.tolist())
                # print(tick)
                for start_idx in start_list.tolist():
                    offsets += [
                        int(start_idx + tick * x) % record.num_frames
                        for x in range(self.num_segments)
                    ]

            return np.array(offsets) + 1
        else:
            raise NotImplementedError("{} not exist".format(self.test_sample))

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder

        full_path = record.path

        while not os.path.exists(full_path):
            print('################## Not Found:',record.path)

            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            full_path = record.path

        if not self.test_mode:
            segment_indices = self._sample_indices(
                record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get_frames(self,path,index):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index-1)
        flag, frame = cap.read()
        frame = frame[:,:,::-1] # BGR2RGB

        return [Image.fromarray(frame).convert('RGB')]


    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                # seg_imgs = self._load_image(record.path, p)
                seg_imgs = self.get_frames(record.path,p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data, label = self.transform((images, record.label))
        return process_data, label

    def __len__(self):
        return len(self.video_list)
