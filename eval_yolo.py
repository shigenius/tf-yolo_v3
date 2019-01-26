# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import time

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image, letter_box_pos_to_original_pos, convert_to_original_size, _iou

import tensorflow.contrib.slim as slim
from nets.vgg import vgg_16, vgg_arg_scope
import cv2
import copy
import config_yolo as cfg
from pathlib import Path
import csv
import os

def get_annotation(image_path, txtname="subwindow_log.txt"):
    img_p = Path(image_path)
    img_obj_name = img_p.parents[1].name
    cropped_dir_p = Path(str(img_p.parent)+'_cropped')
    log_p = cropped_dir_p/txtname
    assert log_p.exists(), 'Does not exist :{0}'.format(str(log_p))

    img_id = int(img_p.stem.split('_')[1])# フレーム番号

    anno = None
    with open(str(log_p), 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if int(row[0]) == img_id:
                anno = row
                break

    return anno # [frame, center_x, center_y, size_x, size_y]

def evaluate(filtered_boxes, gt_anno, orig_img, thresh=0.5):
    tp = 0
    fp = 0
    highest_conf = 0.0
    highest_conf_label = -1

    gt_label = [key for key in gt_anno.keys()][0]

    bbox_list = [] # bbox_dict: [bbox_infomation, cls_id]
    for cls, bboxs in filtered_boxes.items():
        for bbox in bboxs:
            bbox_list.append([bbox, cls])

    iou_l = []
    total_positives = len(bbox_list)
    for [bbox, cls] in bbox_list:
        _bbox = copy.deepcopy(bbox[0])
        orig_scale_bbox = convert_to_original_size(_bbox, np.array((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
                                                   np.array(orig_img.size), True)  # 元のscaleに戻す
        hoge = []
        for gt_bbox in gt_anno.values(): # gtのbbox毎の処理
            iou_ = _iou(orig_scale_bbox,
                       gt_bbox)  # :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
            hoge.append(iou_)
        iou = max(hoge) # 一番大きい値をとる
        iou_l.append(iou)

        if cls == gt_label: # classが一致しているか
            if iou > thresh:
                tp += 1 # labelが一致かつiou>threshの時TPを+1
            else:
                fp += 1
        else:
            fp += 1

        if bbox[1] > highest_conf:
            highest_conf = bbox[1]
            highest_conf_label = cls

    precision = tp / total_positives
    average_iou = sum(iou_l)/(len(iou_l) + 1e-05) # 画像一枚のIoU
    fn = len(gt_anno.values()) - (tp+fp) if len(gt_anno.values()) - (tp+fp) > 0 else 0
    return [tp, fp, fn], average_iou, precision, highest_conf_label

def main(argv=None):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )


    classes = load_coco_names(cfg.CLASS_NAME)

    if cfg.FROZEN_MODEL:
        pass

    else:
        if cfg.TINY:
            model = yolo_v3_tiny.yolo_v3_tiny
        else:
            model = yolo_v3.yolo_v3

        boxes, inputs = get_boxes_and_inputs(model, len(classes), cfg.IMAGE_SIZE, cfg.DATA_FORMAT)
        # boxes : coordinates of top left and bottom right points.
        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

        with tf.Session(config=config) as sess:
            t0 = time.time()
            saver.restore(sess, cfg.CKPT_FILE)
            print('YOLO v3 Model restored in {:.2f}s'.format(time.time()-t0), "from:", cfg.CKPT_FILE)

            # prepare test set
            with open(cfg.TEST_FILE_PATH, 'r') as f:
                f_ = [line.rstrip().split() for line in f]

            data = [[l, get_annotation(l[0], txtname=cfg.GT_INFO_FILE_NAME)] for l in f_]  # data: [[(path_str, label), [frame, center_x, center_y, size_x, size_y]],...]
            data = [l for l in data if l[1] is not None]  # annotationを取得できなかった画像は飛ばす

            def is_cropped_file_Exist(orig_filepath):
                d, file =  os.path.split(orig_filepath)
                cropped_d = d+"_cropped"
                cropped_file = os.path.join(cropped_d, file)
                return os.path.exists(cropped_file)

            data = [l for l in data if is_cropped_file_Exist(l[0][0])] # 対となるcrop画像がない画像は飛ばす

            # log
            f = open(cfg.OUTPUT_LOG_PATH, 'w')
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['image path', 'class/movie_name', 'IoU', 'TP', 'FP', 'FN', 'Average Precision',  'gt label',' highest_conf_label', 'detect time'])

            total_iou = [] # 画像毎のiouのリスト
            total_tp = 0 # TP : IoU > 0.5かつ GT==Pred_classのPositiveの数
            total_fp = 0 # FP : TPの条件以外のPositivesの数
            total_fn = 0 # FN : 検出されなかったGT
            total_ap = []# 画像毎のaverage precisionのリスト．AP : TP / total positives

            # iterative run
            for count, gt in enumerate(data):  # gt: [(path_str, label), [frame, center_x, center_y, size_x, size_y]
                iou = 0.0
                tp = 0
                fp = 0
                fn = 0
                ap = 0.0


                # for evaluation
                gt_box = [float(i) for i in gt[1][1:]]
                gt_box = [gt_box[0] - (gt_box[2] / 2), gt_box[1] - (gt_box[3] / 2), gt_box[0] + (gt_box[2] / 2),
                          gt_box[1] + (gt_box[3] / 2)]
                gt_label = int(gt[0][1]) # GT のクラス
                gt_anno = {gt_label: gt_box}

                print(count, ":", gt[0][0])
                img = Image.open(gt[0][0])
                img_resized = letter_box_image(img, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 128)
                img_resized = img_resized.astype(np.float32)

                t0 = time.time()
                detected_boxes = sess.run(
                    boxes, feed_dict={inputs: [img_resized]})

                filtered_boxes = non_max_suppression(detected_boxes,
                                                     confidence_threshold=cfg.CONF_THRESHOLD,
                                                     iou_threshold=cfg.IOU_THRESHOLD)
                detect_time = time.time()-t0

                print("detected boxes in :{:.2f}s ".format(detect_time), filtered_boxes)

                print(filtered_boxes)
                if len(filtered_boxes.keys()) != 0:  # 何かしら検出された時
                    [tp, fp, fn], iou, precision, highest_conf_label = evaluate(filtered_boxes, gt_anno, img,
                                                                         thresh=0.5)  # 一枚の画像の評価を行う
                else: # 何も検出されなかった時
                    iou = 0.0
                    precision = 0.0
                    tp = 0
                    fp = 0
                    fn = len(gt_anno.values())
                    highest_conf_label = -1

            #
            #     print(filtered_boxes)
            #     if len(filtered_boxes.keys()) != 0:  # 何かしら検出された時
            #         is_detected = True
            #         [tp, fp, fn], iou, ap, highest_conf_label = evaluate(filtered_boxes, gt_anno, img, thresh=0.1) #一枚の画像の評価を行う
            #
            #     else:  # 何も検出されなかった時
            #         is_detected = False
            #         iou = 0.0
            #         ap = 0.0
            #         tp = 0
            #         fp = 0
            #         fn = len(gt_anno.values())
            #         highest_conf_label = -1
            #
                total_iou.append(iou)
                total_ap.append(precision)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                print("IoU:", iou)
                print("mean average IoU:", sum(total_iou)/(len(total_iou) + 1e-05))
                print("AP:", precision)
                print("mAP:", sum(total_ap)/(len(total_ap) + 1e-05))
            #
            #
            #     # draw pred_bbox
            #     draw_boxes(filtered_boxes, img, classes, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), True)
            #     # draw GT
            #     draw = ImageDraw.Draw(img)
            #     color = (0, 0, 0)
            #     draw.rectangle(gt_box, outline=color)
            #     draw.text(gt_box[:2], 'GT_'+classes[gt_label], fill=color)
            #
            #     img.save(os.path.join(cfg.OUTPUT_DIR, '{0:04d}_'.format(count)+os.path.basename(gt[0][0])))

                movie_name = os.path.basename(os.path.dirname(gt[0][0]))
                movie_parant_dir = os.path.basename(os.path.dirname(os.path.dirname(gt[0][0])))
                pred_label = classes[highest_conf_label] if highest_conf_label != -1 else "None"
                save_messe = [gt[0][0], os.path.join(movie_name, movie_parant_dir), iou, tp, fp, fn, precision, classes[gt_label], pred_label, detect_time]
                writer.writerow(save_messe)
                print(save_messe)

            print("total tp :", total_tp)
            print("total fp :", total_fp)
            print("total fn :", total_fn)
            f.close()
            print("proc finished.")


if __name__ == '__main__':
    tf.app.run()
