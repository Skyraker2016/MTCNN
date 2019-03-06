import argparse,cv2
from mtcnn import MTCNN
import os
import random
import time
import math


# video_path: input video's path, .mp4 is allow
# output_path: the floder for imgs, picture will be named as '$output_path$/$video_name$_$index$.jpg', box information in '$output_path$/$video_name$_$index$.txt'
# bbox_path: bbox信息文件路径
# min_size: the bbox's min_size
# accept_rate: only the face got the scores larger than accept_rate is accept
# min_num: min bbox's number 
# max_num: max bbox's number
# frame_gap: 提取帧间隔

def catch_video(video_path, output_path, video_id, bbox_path='./meta.txt', min_size=60, accept_rate=0.99, min_num=1, max_num=3, accept_prob=0.01, frame_gap=-1, begin_frame=0, show=False):
    mtcnn = MTCNN('./mtcnn.pb')
    cap=cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("video fps:", fps)
    if (frame_gap == -1):
        frame_gap = round(fps)
    video_name = str(video_id)
    index = -1
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    while True:
        # 取帧
        index += 1
        ret,img=cap.read()
        if not ret:
            break
        if (begin_frame > index):
            continue

        # # 随机丢帧
        # if random.random() > accept_prob:
        #     continue
        # 按间隔丢帧
        if index % frame_gap != 0:
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

        f = open(bbox_path, 'a')

        # 保存
        time_second = index / fps
        

        save_name = video_name + '_' + get_time_str(time_second)
        print(index, get_time_str(time_second), ' total box:', len(bbox))
        tmp_img = img.copy()
        show_img = None
        for face_i, box in enumerate(bbox):
            box = box.astype('int32')
            # f.write("%s\t%d\t%d\t%d\t%d\n" % (output_path + save_name, box[1], box[0], box[3], box[2]))
            show_img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 3)
            show_img = cv2.putText(img, str(face_i), (box[1], box[0]), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        # if show:
        
        flag = False
        cv2.imshow('img', show_img)
        cv2.waitKey(1)
        for face_i, box in enumerate(bbox):
            emotion = input("face %d: " % (face_i, ))
            try:
                emotion_id = int(emotion)
                f.write("%s\t%d\t%d\t%d\t%d\t%d\n" % (output_path + save_name, box[1], box[0], box[3], box[2], emotion_id))
                flag = True # 如果没有一个表情成功，则不保留该图像
            except:
                print("Not emotion")
                continue
        f.close()
        if flag:
            cv2.imwrite( output_path + save_name+'.jpg', tmp_img)
        # cv2.imwrite(output_path + 'bbox_' + save_name+'.jpg', show_img)



def get_time_str(second):
    int_second = math.floor(second)
    ms = math.floor((second - int_second) * 100)
    s = int_second % 60
    m = int_second // 60 % 60
    h = int_second // 3600
    return "%02d%02d%02d%02d" % (h, m, s, ms)



if __name__ == '__main__':
    catch_video('ep40.mp4', './100040/', 100040)
    # print(get_time_str(142.88))