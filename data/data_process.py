import cv2
from imutils import contours
import numpy as np



def segment(result,cntrs,saveToPath,name):
    import os
    tempImage=result.copy()
    list_of_digits=[]
    for i,c in enumerate(cntrs):
        area = cv2.contourArea(c)
        min_area_threshold = 500 
        if area > min_area_threshold:
            box = cv2.boundingRect(c)
            x,y,w,h = box
            temp=tempImage[y:y+h,x:x+w]
            padded_digit = np.pad(temp, ((25,25),(25,25),(0,0)), "constant", constant_values=0)
            cv2.imwrite(os.path.join(saveToPath,name+f"{i+1}.png"),padded_digit)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
            list_of_digits.append(padded_digit)
    return list_of_digits


def makeContours(temp,kernalSize):
    gray=cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernalSize)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    cntrs = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
    return cntrs

def clearPath(path):
    import os
    import shutil
    for files in os.listdir(path):
        filepath = os.path.join(path, files)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)
    return

def split_digit_from_img(path):
    import os
    from imutils import contours
    img=cv2.imread(path["images"])
    cropped=img[20:img.shape[0]-20,30:img.shape[1]-20]
    cntrs=makeContours(img,(150,3))
    (cnts, _) = contours.sort_contours(cntrs, method="left-to-right")
    result=segment(img,cnts,path["lines"],"line")
    return result


# path = {
#     "images": "data/test/digit_test.png",
#     "lines": "data/lines/",
#     "words": "data/words/",
#     "letter": "data/letter/"
# }

# split_digit_from_img(path)