import argparse,cv2
from mtcnn import MTCNN
import os
import random


# video_path: input video's path, .mp4 is allow
# output_path: the floder for imgs, picture will be named as '$output_path$/$video_name$_$index$.jpg', box information in '$output_path$/$video_name$_$index$.txt'
# min_size: the bbox's min_size
# accept_rate: only the face got the scores larger than accept_rate is accept
# min_num: min bbox's number 
# max_num: max bbox's number
# accept_prob: 每个图的接受概率（用于减少重复帧）

def catch_video(video_path, output_path, min_size=60, accept_rate=0.99, min_num=1, max_num=3, accept_prob=0.01, show=False):
    mtcnn = MTCNN('./mtcnn.pb')
    cap=cv2.VideoCapture(video_path)
    video_name = video_path.split('/')[-1]
    index = -1
    # f = open( output_path + 'bbox' + '.txt', 'a')
    while True:
        # 取帧
        index += 1
        ret,img=cap.read()
        if not ret:
            break
        # 随机丢帧
        if random.random() > accept_prob:
            continue

        # 筛选帧
        bbox, scores, landmarks = mtcnn.detect(img)
        new_bbox = []
        for box, score in zip(bbox, scores):
            if score < accept_rate:
                continue
            new_bbox.append(box)

        bbox = new_bbox
        if (len(bbox) < min_num or len(bbox) > max_num):
            continue

        f = open(output_path + 'bbox' + '.txt', 'a')

        # 保存
        save_name = video_name + '_' + str(index)
        print(index, ' total box:', len(bbox))
        cv2.imwrite( output_path + save_name+'.jpg', img)
        show_img = None
        for face_i, box in enumerate(bbox):
            box = box.astype('int32')
            f.write("%s\t%d\t%d\t%d\t%d\n" % (output_path + save_name, box[1], box[0], box[3], box[2]))
            show_img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 3)
            show_img = cv2.putText(img, str(face_i), (box[1], box[0]), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        if show:
            cv2.imshow('img', show_img)
            cv2.waitKey(1)
        f.close()
        cv2.imwrite(output_path + 'bbox_' + save_name+'.jpg', show_img)
    f.close()

if __name__ == '__main__':
    catch_video('ep40.mp4', './target/')