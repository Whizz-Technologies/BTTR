import cv2
import numpy as np
import base64
import sys

from scan import DocScanner
from bttr.lit_bttr import LitBTTR
from torchvision.transforms import ToTensor
ckp_path = "./pretrained-2014.ckpt"
model = LitBTTR.load_from_checkpoint(ckp_path)
import cv2
def crop_image(img):
    #find the heighest white pixel in the image
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #convert image to binary
    #img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    _,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    h, w = thresh.shape
    height = 100
    r = height/thresh.shape[0]
    dim = ((int(thresh.shape[1]*r), height))
    thresh = cv2.resize(thresh,(dim))
    h, w = thresh.shape
    #cv2.imshow('thres', thresh)
    #cv2.waitKey(0)
    print(thresh.shape)
    max_white_h = 0
    max_white_w = 0
    for i in range(h):
        for j in range(w):
            if thresh[i][j] == 255:
                if(max_white_h < i):
                    max_white_h = i

                if(max_white_w < j):
                    max_white_w = j

    print("Max position", max_white_w, max_white_h)

    #Print the lowest white pixel in the image
    min_white_h = h
    min_white_w = w
    for i in range(h):
        for j in range(w):
            if thresh[i][j] == 255:
                if(min_white_h > i):
                    min_white_h = i

                if(min_white_w > j):
                    min_white_w = j

    print("Minumum position", min_white_w, min_white_h)

    #crop image
    #k = thresh[min_white_h:max_white_h, min_white_w:max_white_w]
    if(min_white_h - 5 < 0):
        min_white_h = 0
    if(min_white_w - 5 < 0):
        min_white_w = 0
    if(max_white_h + 5 > h):
        max_white_h = h
    if(max_white_w + 5 > w):
        max_white_w = w
    k = thresh[min_white_h-5:max_white_h+5, 0:w]
    #k = thresh[min_white_h:max_white_h, min_white_w:max_white_w]
    #k = thresh[min_white_h+5:max_white_h-5, min_white_w+5:max_white_w-5]

    #print(k)
    cv2.imshow('img', k)
    cv2.waitKey(0)
    return k

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou



path = './test/test3.jpeg'
#image = cv2.imread('./test/test1.jpeg')
scanner = DocScanner(True)
image = scanner.scan(path)
#cv2.imshow('image',image)
#cv2.waitKey(0)
#gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#Blur=cv2.GaussianBlur(gray_image,(5,5),1) #apply blur to roi
#Canny=cv2.Canny(Blur,10,50) #apply canny to roi
#_, thrash = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
ret3,thrash = cv2.threshold(image,5,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.imshow("thrash",thrash)
#cv2.waitKey(0)
#Find my contours
contours,_ = cv2.findContours(thrash,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
#cv2.imshow("image",image)
#cv2.waitKey(0)
#cv2.imwrite("./cnt.jpeg",image)
cntrRect = []
rect_list = []
for contour in contours:

    epsilon = 0.01*cv2.arcLength(contour,True)

    approx = cv2.approxPolyDP(contour,epsilon,True)
    #cv2.drawContours(img, [approx], 0, (0,0,0), 5)

    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if len(approx) == 4:
      #print("Rect")
      x, y, w, h = cv2.boundingRect(approx)
      dict_rect = {"x" : x , "y" : y , "w" : w , "h" : h}
      print('area')
      print(w*h)
      #print(dict_rect)
      if((w*h > 1000) and (w*h < 100000)):
        dict_rect_copy = dict_rect.copy()
        rect_list.append(dict_rect_copy)
      #print(rect_list)
      aspectRatio = float(w)/h
      #print(aspectRatio)
      print("X", x,"Y",y,"W",w,"H",h)
      #cv2.drawContours(image, [approx], 0, (0,0,0), 1)
      #cv2.putText(image, "Rectrangle", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
      #cv2.imwrite("./rect.jpeg", image)

#cv2.imshow("image",img)
#cv2.waitKey(0)
new_list = []
for box in rect_list:
  x = box['x']
  y = box['y']
  w = box['w']
  h = box['h']
  bb1 = (x,y,x+w,y+h)
  k = 1
  #if list is empty
  if len(new_list) == 0:
    new_list.append(box)
  else:
    for box2 in new_list:
      x2 = box2['x']
      y2 = box2['y']
      w2 = box2['w']
      h2 = box2['h']
      bb2 = (x2,y2,x2+w2,y2+h2)
      iou_value = iou(bb1,bb2)
      #print(iou_value)
      if iou_value > 0.5:
        k = 0
        break
  if(k == 1):
    print(box)
    new_list.append(box)

for bb in new_list:
  x = bb['x']
  y = bb['y']
  w = bb['w']
  h = bb['h']
  print(".....................")
  print(x,y,w,h)
  '''if( y > (y+h)):
    continue
  elif( x > (x+w)):
    continue
  #elif((x+w-50) > 1000):
  #  continue
  elif((h) > 1000):
    continue'''
  cropped_image = image[y+10:(y+h-10), x+10:(x+w-10)]
  #cropped_image = image[y:(y+h), x:(x+w)]
  l = crop_image(img=cropped_image)
  #print(cropped_image.shape)
  img = ToTensor()(l)
  hyp = model.beam_search(img)
  print(hyp)