import cv2

def segment(result,cntrs,saveToPath,name):
    import os
    tempImage=result.copy()
    for i,c in enumerate(cntrs):
        box = cv2.boundingRect(c)
        x,y,w,h = box
        temp=tempImage[y:y+h,x:x+w]
        cv2.imwrite(os.path.join(saveToPath,name+f"{i}.png"),temp)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return result


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

def main(path):
    import os
    img=cv2.imread(path["images"])
    # cropped=img[20:img.shape[0]-20,30:img.shape[1]-20]
    contours=makeContours(img,(150,3))
    # clearPath(path["lines"])
    result=segment(img,contours,path["lines"],"line")
    # clearPath(path["words"])
    count=0
    for file in sorted(os.listdir(path["lines"])):
        filePath=os.path.join(path["lines"],file)
        line=cv2.imread(filePath)
        name=f"line{count}word"
        count+=1
        contours=makeContours(line,(3,150))
        resultLine=segment(line,contours,path["words"],name)
    for file in os.listdir(path["words"]):
        filePath=os.path.join(path["words"],file)
        word=cv2.imread(filePath)
        name=file.split('.')[0]+"letter"
        contours=makeContours(word,(1,2))
        resultWord=segment(word,contours,path["letter"],name)
    clearPath(path["letter"])
    cv2.imshow("Line segmentation",result)#uncomment this for viewing the line segmentation
    cv2.waitKey(0)

path = {
    "images": "data/test/digit_test.png",
    "lines": "data/lines/",
    "words": "data/words/",
    "letter": "data/letter/"
}

main(path)