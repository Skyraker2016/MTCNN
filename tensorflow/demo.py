import argparse,cv2
from mtcnn import MTCNN

def test_image(imgpath):
    mtcnn = MTCNN('./mtcnn.pb')
    img = cv2.imread(imgpath)

    bbox, scores, landmarks = mtcnn.detect(img)

    print('total box:', len(bbox))
    for box, pts in zip(bbox, landmarks):
        box = box.astype('int32')
        img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 3)

        pts = pts.astype('int32')
        for i in range(5):
            img = cv2.circle(img, (pts[i+5], pts[i]), 1, (0, 255, 0), 2)
    cv2.imshow('image', img)
    cv2.waitKey()

def test_camera(index=0):
    mtcnn = MTCNN('./mtcnn.pb')
    cap=cv2.VideoCapture(index)
    while True:
        ret,img=cap.read()
        if not ret:
            break
        bbox, scores, landmarks = mtcnn.detect(img)
        print('total box:', len(bbox))
        for box, pts in zip(bbox, landmarks):
            box = box.astype('int32')
            img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 3)
            pts = pts.astype('int32')
            for i in range(5):
                img = cv2.circle(img, (pts[i+5], pts[i]), 1, (0, 255, 0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    #test_image()
    test_camera()