"""
https://github.com/xingyizhou/CenterTrack
Modified by Rufeng Zhang
"""
import os
import numpy as np
import json
import cv2

# Use the same script for MOT16
DATA_PATH = '/home/xiongp/datasets/airplaneyolovid'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
SPLITS = ['val2017']  # --> split training data to train_half and val_half.
HALF_VIDEO = False
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True

severalDataset = ['data22']

if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:

        data_path = os.path.join(DATA_PATH, split)
        # out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out_path = os.path.join(OUT_PATH, '{}.json'.format("val"))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 1, 'name': 'airplane'},
                              {'id': 2, 'name': 'none'}]}
        seqs = os.listdir(data_path)
        image_cnt = 0
        video_cnt = 0
        ann_cnt = 0
        for seq in sorted(seqs):
            if (not 'data' in seq) or ('datav' in seq) or ('datag' in seq) or ('data16' in seq) or ('data17' in seq):
                # if not seq in severalDataset:
                continue
            video_cnt += 1  # video sequence number.
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(data_path, seq)
            # gt path
            ann_path = os.path.join(seq_path, 'gt/gt.txt')
            images = os.listdir(seq_path)
            num_images = len([image for image in images if 'jpg' in image])  # half and half

            image_range = [0, num_images - 1]

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                height, width = 256, 256
                image_info = {'file_name': '{}/{:}.jpg'.format(seq, i),  # image name.
                              'id': image_cnt + i ,  # image number in the entire training set.
                              'frame_id': i - image_range[0],
                              # image number in the video sequence, starting from 1.
                              'video_id': video_cnt,
                              'height': height, 'width': width}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))
            for i in range(num_images):
                labels_path = os.path.join(DATA_PATH, 'labels', seq, '{}.txt'.format(i))
                with open(labels_path) as l:
                    lines = l.readlines()
                    if len(lines) < 1:
                        continue
                    track_id = 0
                    for line_ori in lines:
                        line = line_ori.split()
                        for j in range(len(line)):
                            line[j] = (int)(float(line[j]))

                        labels = line
                        labels[3] = line[3] - line[1]
                        labels[4] = line[4] - line[2]
                        frame_id = i
                        track_id += 1
                        ann_cnt += 1
                        category_id = 1
                        ann = {'id': ann_cnt,
                               'category_id': category_id,
                               'image_id': image_cnt + frame_id + 1,
                               'track_id': track_id,
                               'bbox': labels[1:5],
                               'conf': 1,
                               'iscrowd': 0,
                               'area': labels[3] * labels[4]}
                        out['annotations'].append(ann)
            image_cnt += num_images
    print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))
