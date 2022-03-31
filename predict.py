from bttr.lit_bttr import LitBTTR
from torchvision.transforms import ToTensor
ckp_path = "./pretrained-2014.ckpt"
model = LitBTTR.load_from_checkpoint(ckp_path)
import cv2

img = cv2.imread("./Unstructured__Handwritten_Math_Test_2.jpeg")
#convert image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#convert image to binary
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
_,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
#resize while maintaing aspect ratio using opencv
height = 100
r = height/thresh.shape[0]
dim = ((int(thresh.shape[1]*r), height))
thresh = cv2.resize(thresh,(dim))
cv2.imshow('thres', thresh)
cv2.waitKey(0)


def crop_image(img):
    #find the heighest white pixel in the image
    h, w = thresh.shape
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
    k = thresh[min_white_h-5:max_white_h+5, min_white_w-5:max_white_w+5]
    print(k)
    cv2.imshow('img', k)
    cv2.waitKey(0)
    return k

k = crop_image(thresh)
img = ToTensor()(k)
hyp = model.beam_search(img)
print(hyp)
