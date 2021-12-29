import cv2 as cv
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
import math
import os
import random

'''
image = cv.imread("taulell2.jpg")
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray2 = cv.imread("taulell2.jpg",0)


print("ancho: "+str(image.shape[1]))
#print("alto: "+str(image.shape[0]))
#print("canales: "+str(image.shape[2]))
cv.imshow("taulell grayscale", image_gray)
cv.imshow("taulell grayscale2", image_gray2)
cv.imshow("taulell",image)
cv.waitKey(0)
cv.destroyAllWindows()
'''

#def retallar_taulell(img):
    #corners = cv.findChessboardCorners(img,cv.size(11,11), cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)
    #return corners
    #print(type(corners))
    #print(corners)

def interseccions(img):
    
    corners = cv.goodFeaturesToTrack




def resize(img, alto, interpolation=cv.INTER_CUBIC ):
    image = cv.resize(img,( int(alto*(img.shape[1]/img.shape[0])),int(alto)),interpolation)
    return image

def hlines2(img):
    image = resize(img,500)
    
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_canny = cv.Canny(image_gray,150,255)
    _, image_threshold = cv.threshold(image_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #image_blur = cv.GaussianBlur(image_gray,(5,5),0)
    #image_threshold2 = cv.adaptiveThreshold(image_gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
    #_, image_threshold3 = cv.threshold(image_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
   
    
    lines = cv.HoughLinesP(image_canny,0.5,1*np.pi / 180, 78,None, image.shape[1]/2,100)
    #lines = [line for line in lines if line[0][1] > 75*((2*math.pi)/360) and line[0][1] < 105*((2*math.pi)/360)]
    lines2 = []
    for i in lines:
        x1,y1,x2,y2 = i[0]
        angle = -math.atan((y1-y2)/(x1-x2))
        if angle > -10*((2*math.pi)/360) and angle < 10*((2*math.pi)/360):
            lines2.append(i)
            
    for i in lines2:
        x1,y1,x2,y2 = i[0]
        cv.line(image,(x1,y1),(x2,y2),(0,255,0))
        

    cv.imshow('image', image)
    #cv.imshow('image_gray', image_gray)
    cv.imshow('image_canny', image_canny)
    #cv.imshow('image_threshold', image_threshold)
    #cv.imshow('image_threshold2',image_threshold2)
    #cv.imshow('image_threshold3',image_threshold3)
    cv.waitKey(0)
    cv.destroyAllWindows()



path = 'C:/Users/alexn/Desktop/TR/images/Taulell'
taulells = os.listdir(path)
img = cv.imread(path+'/'+str(taulells[random.randint(0,len(taulells)-1)]))
hlines2(img)
img = cv.imread(path+'/'+str(taulells[random.randint(0,len(taulells)-1)]))
hlines2(img)
img = cv.imread(path+'/'+str(taulells[random.randint(0,len(taulells)-1)]))
hlines2(img)

