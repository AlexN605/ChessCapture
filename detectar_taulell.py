import cv2 as cv
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
import math
import sympy as sp
import os
import random
import pandas as pd
import operator


def resize(img, alto, interpolation=cv.INTER_CUBIC ):
    image = cv.resize(img,( int(alto*(img.shape[1]/img.shape[0])),int(alto)),interpolation)
    return image

def show(img):
    cv.imshow('',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def dibuixar_linies(img,linies):
    for i in linies:
        #Passar de coordenades polars a cartesianes i trobar punts en coordenades cartesianes:
        #(x,y) = (d*cos(B),d*sin(B))+K(-sin(B),cos(B))
        #On en coordenades polars:
        # d = rho -> distancia 
        # B = theta -> angle
        d, angle = i
        x = d * math.cos(angle) 
        y = d * math.sin(angle)
        sin = math.sin(angle)
        cos = math.cos(angle)
        cv.line(img,(round(x+2*img.shape[1]*(-sin)),round(y+2*img.shape[0]*cos)),(round(x-2*img.shape[1]*(-sin)),round(y-2*img.shape[0]*cos)),(0,255,255))
    
    return img
        

def corners(img): #Función que detecta las esquinas de un tablero
    img = resize(img,600)
    image_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_equalized = cv.equalizeHist(image_gray)
    image_equalized2 = cv.equalizeHist(image_equalized)
    _, image_threshold = cv.threshold(image_equalized,140,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    corners = cv.goodFeaturesToTrack(image_equalized,200,0.01,image_gray.shape[1]*2/(3*14))
    for i in corners:
        x,y = i[0]
        cv.circle(img,(int(x),int(y)),3,(0,0,255),-1)
    
    #print(corners)    
    # cv.imshow('corners',img)
    # cv.imshow('image_equalized',image_equalized2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return [corners,image_equalized.shape]


def lines_corners(corners,shape):
    negre = np.zeros(shape, np.uint8)
    negre_corners = negre

    # for i in corners:
    #     negre_corners[int(i[0][1])][int(i[0][0])] = float(255)
    #----------------Dibuixar els punts:-------
    for i in corners:
        x,y = i[0]
        cv.circle(negre_corners, (int(x),int(y)),2,255,-1)
    #------------Detectar les linies:-----------------
    lines = cv.HoughLines(negre_corners,0.1,0.5*np.pi / 180,14)
    lines3 = []
    for i in range(len(lines)):
        lines3.append([lines[i][0][0],lines[i][0][1]])
    #-------------------Filtrar per angles----------------------
    lines2 = []
    for i in lines3:
        d, angle = i
        if (angle > 85*((2*math.pi)/360) and angle < 95*((2*math.pi)/360)):
            lines2.append(i)
    
    angles = []
    for i in lines2:
        angles.append(i[1])
    mitja = sum(angles)/len(angles)
    eliminar = []
    for i in range(len(angles)):
        if angles[i] > mitja+mitja*0.01 or angles[i] < mitja-mitja*0.01:
            eliminar.append(i)
    
    if len(eliminar) >= 1:
        eliminar.reverse()
        for i in eliminar:
            lines2.pop(i)
    
    
    #------------Linies solapades:------------------------
    
    lines2 = sorted(lines2, key = operator.itemgetter(0))
    eliminar = []
    for i in range(len(lines2)-1):
        if lines2[i+1][0]-lines2[i][0] <= 10 :#and lines2[i][1] == lines2[i+1][1]
            eliminar.append(i)
    
    if len(eliminar) >= 1:
        eliminar.reverse()
        for i in eliminar:
            lines2.pop(i)
    
    # negre_corners2 = dibuixar_linies(negre_corners,lines)
    
    # cv.imshow('negre_corners',negre_corners)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return lines2


def hlines(img): #Función que detecta las líneas horizontales
    image = resize(img,600)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # mean = round(np.mean(image_gray))
    # image_canny = cv.Canny(image_gray,mean,2*mean)
    image_canny = cv.Canny(image_gray,170,260)
    # _, image_threshold = cv.threshold(image_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # image_blur = cv.GaussianBlur(image_gray,(3,3),0)
    #image_threshold2 = cv.adaptiveThreshold(image_gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
    #_, image_threshold3 = cv.threshold(image_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)




    lines = cv.HoughLines(image_canny,0.4,0.2*np.pi / 180, 75)
    lines3 = []
    for i in range(len(lines)):#treure el doble parentesis(llista dintre d'una altre llista)
        lines3.append([lines[i][0][0],lines[i][0][1]])
        
        
    #-------------- Filtració de linies ----------------------------------
    #-----------------Angle:------------------------------------------
    
    lines2 = []
    for i in lines3:
        d, angle = i
        #lines = [line for line in lines if line[0][1] > 75*((2*math.pi)/360) and line[0][1] < 105*((2*math.pi)/360)]
        if angle > 80*((2*math.pi)/360) and angle < 100*((2*math.pi)/360):
            lines2.append(i)
    
    #---------------Linies solapades----------------------------------
    
    lines2 = sorted(lines2, key = operator.itemgetter(0))
    eliminar = []
    for i in range(len(lines2)-1):
        if lines2[i+1][0]-lines2[i][0] <= 3 :#and lines2[i][1] == lines2[i+1][1]
            eliminar.append(i)
    
    if len(eliminar) >= 1:
        eliminar.reverse()
        for i in eliminar:
            lines2.pop(i)
    #----------------------------------------------------------------------
    # print('Linies h eliminades per linies solapades: {}'.format(len(eliminar)))
    
    #-----------------Angle:------------------------------------------
    
    
    angles = []
    for i in lines2:
        angles.append(i[1])
    mitja_ang = sum(angles)/len(angles)
    eliminar = []
    for i in range(len(angles)):
        if angles[i] > mitja_ang+mitja_ang*0.05 or angles[i] < mitja_ang-mitja_ang*0.05:
            eliminar.append(i)
    if len(eliminar) >= 1:
        for i in eliminar:
            lines2.pop(i)        
    #----------------------------------------------------------------------
    # print('Linies h eliminades per angle: {}'.format(len(eliminar)))
    
    #----------Calcular mitja angles un altre cop---------     
    angles = []
    for i in lines2:
        angles.append(i[1])
    mitja_ang = sum(angles)/len(angles)
    
    #---------------------Distancia entre linies---------------------------
    distancies = []
    for i in range(len(lines2)-1):
        distancies.append(lines2[i+1][0] - lines2[i][0])
    mitja_dist = sum(distancies)/len(distancies)
    eliminar = []
    for i in range(len(lines2)-1):
        if lines2[i+1][0]-lines2[i][0] < mitja_dist/2:
            if i == len(lines2)-2:
                if lines2[i][0]-lines2[i-1][0] < mitja_dist+mitja_dist*0.05 and lines2[i][0]-lines2[i-1][0] > mitja_dist-mitja_dist*0.05:
                    if lines2[i+1][0]-lines2[i-1][0] < mitja_dist+mitja_dist*0.05 and lines2[i+1][0]-lines2[i-1][0] > mitja_dist-mitja_dist*0.05:
                        if abs(lines2[i][1]-mitja_ang) < abs(lines2[i+1][1]-mitja_ang):
                            eliminar.append(i+1)
                        else:
                            eliminar.append(i)
                    else:
                        eliminar.append(i+1)
                elif lines2[i+1][0]-lines2[i-1][0] < mitja_dist+mitja_dist*0.05 and lines2[i+1][0]-lines2[i-1][0] > mitja_dist-mitja_dist*0.05:
                    eliminar.append(i)
                
                else:
                    if abs(lines2[i][1]-mitja_ang) < abs(lines2[i+1][1]-mitja_ang):
                        eliminar.append(i+1)
                    else:
                        eliminar.append(i)
            else:
                if lines2[i+2][0]-lines2[i][0] < mitja_dist+mitja_dist*0.05 and lines2[i+2][0]-lines2[i][0] > mitja_dist-mitja_dist*0.05:
                    if lines2[i+2][0]-lines2[i+1][0] < mitja_dist+mitja_dist*0.05 and lines2[i+2][0]-lines2[i+1][0] > mitja_dist-mitja_dist*0.05:
                        if abs(lines2[i][1]-mitja_ang) < abs(lines2[i+1][1]-mitja_ang):
                            eliminar.append(i+1)
                        else:
                            eliminar.append(i)
                    else:
                        eliminar.append(i+1)
                elif lines2[i+2][0]-lines2[i+1][0] < mitja_dist+mitja_dist*0.05 and lines2[i+2][0]-lines2[i+1][0] > mitja_dist-mitja_dist*0.05:
                    eliminar.append(i)
                
                else:
                    if abs(lines2[i][1]-mitja_ang) < abs(lines2[i+1][1]-mitja_ang):
                        eliminar.append(i+1)
                    else:
                        eliminar.append(i)
    
    if len(eliminar) >= 1:
        eliminar.reverse()
        for i in eliminar:
            lines2.pop(i)
    # ----------------------------------------------------------------------
    print('Linies h eliminades per distancia: {}'.format(len(eliminar)))
    
    # ------------------Eliminar sobras------------------
    if len(lines2) <= 9:
        check = False
    else:
        check = True
        
    while check:
        if len(lines2) == 10:
            if lines2[0][0] <= image.shape[0] - lines2[-1][0]:
                lines2.pop(0)
                check = False
            else:
                lines2.pop(-1)
                check = False
            
        elif len(lines2) == 11:
            lines2.pop(-1)
            lines2.pop(0)
            check = False
        
        elif len(lines2) == 12:
            if lines2[0][0] <= image.shape[0] - lines2[-1][0]:
                lines2.pop(0)
            else:
                lines2.pop(-1)
        
        elif len(lines2) > 12:
            lines2.pop(-1)
            lines2.pop(0)

        
    
    
    # cv.imshow('hlines', image)
    #cv.imshow('image_gray', image_gray)
    # cv.imshow('image_canny', image_canny)
    # cv.imshow('image_blur', image_blur)
    # cv.imshow('image_threshold', image_threshold)
    #cv.imshow('image_threshold2',image_threshold2)
    #cv.imshow('image_threshold3',image_threshold3)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    return lines2

def vlines(img): #Función que detecta las líneas verticales
    image = resize(img,600)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_canny = cv.Canny(image_gray,120,260)
    
    lines = cv.HoughLines(image_canny,0.4,0.2*np.pi / 180, 70)
    lines3 = []
    for i in range(len(lines)):#treure el doble parentesis(llista dintre d'una altre llista)
        lines3.append([lines[i][0][0],lines[i][0][1]])
    
    #-------------- Filtració de linies ----------------------------------
    #-----------------Angle:------------------------------------------
    
    lines2 = []
    for i in lines3:
        d, angle = i
        if (angle > 350*((2*math.pi)/360) or angle < 10*((2*math.pi)/360)):
            lines2.append(i)
    #---------------Linies solapades----------------------------------

    
    lines2 = sorted(lines2, key = operator.itemgetter(0))
    eliminar = []
    for i in range(len(lines2)-1):
        if lines2[i+1][0]-lines2[i][0] <= 3 :#and lines2[i][1] == lines2[i+1][1]
            eliminar.append(i)
    
    if len(eliminar) >= 1:
        eliminar.reverse()
        for i in eliminar:
            lines2.pop(i)
    #----------------------------------------------------------------------
    # print('Linies v eliminades per linies solapades: {}'.format(len(eliminar)))
    
    
    #-----------------Angle:------------------------------------------
    
    angles = []
    for i in lines2:
        angles.append(i[1])
    mitja_ang = sum(angles)/len(angles)
    
    eliminar=[]
    for i in range(len(angles)):
        if abs(angles[1]-mitja_ang) > 4*math.pi/180:
            eliminar.append(i)
    
    
    if len(eliminar) >= 1:
        eliminar.reverse()
        for i in eliminar:
            lines2.pop(i)
    #----------------------------------------------------------------------
    # print('Linies v eliminades per angle: {}'.format(len(eliminar)))

    
    #----------Calcular mitja angles un altre cop---------     
    angles = []
    for i in lines2:
        angles.append(i[1])
    mitja_ang = sum(angles)/len(angles)
    
    #---------------------Distancia entre linies---------------------------
    distancies = []
    for i in range(len(lines2)-1):
        distancies.append(lines2[i+1][0] - lines2[i][0])
    mitja_dist = sum(distancies)/len(distancies)
    eliminar = []
    for i in range(len(lines2)-1):
        if lines2[i+1][0]-lines2[i][0] < mitja_dist/2:
            if i == len(lines2)-2:
                if lines2[i][0]-lines2[i-1][0] < mitja_dist+mitja_dist*0.05 and lines2[i][0]-lines2[i-1][0] > mitja_dist-mitja_dist*0.05:
                    if lines2[i+1][0]-lines2[i-1][0] < mitja_dist+mitja_dist*0.05 and lines2[i+1][0]-lines2[i-1][0] > mitja_dist-mitja_dist*0.05:
                        if abs(lines2[i][1]-mitja_ang) < abs(lines2[i+1][1]-mitja_ang):
                            eliminar.append(i+1)
                        else:
                            eliminar.append(i)
                    else:
                        eliminar.append(i+1)
                elif lines2[i+1][0]-lines2[i-1][0] < mitja_dist+mitja_dist*0.05 and lines2[i+1][0]-lines2[i-1][0] > mitja_dist-mitja_dist*0.05:
                    eliminar.append(i)
                
                else:
                    if abs(lines2[i][1]-mitja_ang) < abs(lines2[i+1][1]-mitja_ang):
                        eliminar.append(i+1)
                    else:
                        eliminar.append(i)
            else:
                if lines2[i+2][0]-lines2[i][0] < mitja_dist+mitja_dist*0.05 and lines2[i+2][0]-lines2[i][0] > mitja_dist-mitja_dist*0.05:
                    if lines2[i+2][0]-lines2[i+1][0] < mitja_dist+mitja_dist*0.05 and lines2[i+2][0]-lines2[i+1][0] > mitja_dist-mitja_dist*0.05:
                        if abs(lines2[i][1]-mitja_ang) < abs(lines2[i+1][1]-mitja_ang):
                            eliminar.append(i+1)
                        else:
                            eliminar.append(i)
                    else:
                        eliminar.append(i+1)
                elif lines2[i+2][0]-lines2[i+1][0] < mitja_dist+mitja_dist*0.05 and lines2[i+2][0]-lines2[i+1][0] > mitja_dist-mitja_dist*0.05:
                    eliminar.append(i)
                
                else:
                    if abs(lines2[i][1]-mitja_ang) < abs(lines2[i+1][1]-mitja_ang):
                        eliminar.append(i+1)
                    else:
                        eliminar.append(i)
    
    if len(eliminar) >= 1:
        eliminar.reverse()
        for i in eliminar:
            lines2.pop(i)
    #----------------------------------------------------------------------
    # print('Linies v eliminades per distancia: {}'.format(len(eliminar)))
   
    #------------------Eliminar sobrantes------------------
    
    if len(lines2) <= 9:
        check = False
    else:
        check = True
        
    while check:
        if len(lines2) == 10:
            if lines2[0][0] <= image.shape[1] - lines2[-1][0]:
                lines2.pop(0)
                check = False
            else:
                lines2.pop(-1)
                check = False
            
        elif len(lines2) == 11:
            lines2.pop(-1)
            lines2.pop(0)
            check = False
        
        elif len(lines2) == 12:
            if lines2[0][0] <= image.shape[1] - lines2[-1][0]:
                lines2.pop(0)
            else:
                lines2.pop(-1)
        
        elif len(lines2) > 12:
            lines2.pop(-1)
            lines2.pop(0)
    
    
    
    # cv.imshow('image_canny', image_canny)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
        
    
    return lines2

def interseccions(hlines, vlines): #Función que calcula las intersecciones entre las líneas verticales y las horizontales
    equacions_h = []
    equacions_v = []
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    for i in hlines:
        d, angle = i
        if angle == 0:
            equacions_h.append(d-x)
        
        elif angle > 0 and angle < np.pi/2:
            m = -1/(math.tan(angle))
            n = d/math.cos((np.pi/2)-angle)
            equacions_h.append(m*x-y+n)
            
        elif angle == np.pi/2:
            equacions_h.append(d-y)
            
        elif angle > np.pi/2 and angle < np.pi:
            m = -1/(math.tan(angle))
            n = d/math.cos(angle-np.pi/2)
            equacions_h.append(m*x-y+n)
        
        elif angle == np.pi:
            equacions_h.append(-d-x)
        
        elif angle > np.pi and angle < 3*np.pi/2:
            m = -1/(math.tan(angle-np.pi))
            n = d/math.cos(3*np.pi/2-angle)
            equacions_h.append(m*x-y+n)        
        
        elif angle == 3*np.pi/2:
            equacions_h.append(-d-y)
        
        elif angle > 3*np.pi/2 and angle < 2*np.pi:
            m = -1/(math.tan(angle-np.pi))
            n = d/math.cos(angle-3*np.pi/2)
            equacions_h.append(m*x-y+n)
        
        elif angle == 2*np.pi:
            equacions_h.append(d-x)
    
    for i in vlines:
        d, angle = i
        if angle == 0:
            equacions_v.append(d-x)
        
        elif angle > 0 and angle < np.pi/2:
            m = -1/(math.tan(angle))
            n = d/math.cos((np.pi/2)-angle)
            equacions_v.append(m*x-y+n)
            
        elif angle == np.pi/2:
            equacions_v.append(d-y)
            
        elif angle > np.pi/2 and angle < np.pi:
            m = -1/(math.tan(angle))
            n = d/math.cos(angle-np.pi/2)
            equacions_v.append(m*x-y+n)
        
        elif angle == np.pi:
            equacions_v.append(-d-x)
        
        elif angle > np.pi and angle < 3*np.pi/2:
            m = -1/(math.tan(angle-np.pi))
            n = d/math.cos(3*np.pi/2-angle)
            equacions_v.append(m*x-y+n)        
        
        elif angle == 3*np.pi/2:
            equacions_v.append(-d-y)
        
        elif angle > 3*np.pi/2 and angle < 2*np.pi:
            m = -1/(math.tan(angle-np.pi))
            n = d/math.cos(angle-3*np.pi/2)
            equacions_v.append(m*x-y+n)
        
        elif angle == 2*np.pi:
            equacions_v.append(d-x)
    
    #-------------Solucionar equacions----------
    punts = [[],[],[],[],[],[],[],[],[]]
    for l in range(len(equacions_h)):
        for c in range(len(equacions_v)):
            sol = sp.solve([equacions_h[l],equacions_v[c]],x,y)
            punts[l].append([int(round(sol[x])),int(round(sol[y]))])
    
    return punts
           
        
        
        
path = 'C:/Users/alexn/Desktop/TR/images/Taulells3'
taulells = os.listdir(path)
'''
img = cv.imread(path+'/'+str(taulells[random.randint(0,len(taulells)-1)]))
hlines(img)

img = cv.imread(path+'/'+str(taulells[random.randint(0,len(taulells)-1)]))
hlines(img)
img = cv.imread(path+'/'+str(taulells[random.randint(0,len(taulells)-1)]))
hlines(img)
'''
num_imatge = 26
img = cv.imread(path+'/'+str(taulells[num_imatge]))
img = resize(img,600)
img2 = cv.imread(path+'/'+str(taulells[num_imatge]))
img2 = resize(img,600)

# random_num = random.randint(0,len(taulells)-1)
# print('Imatge numero {}'.format(random_num))
# print('Nom de la imatge: {}'.format(taulells[random_num]))
# img = cv.imread(path+'/'+str(taulells[random_num]))
# img = resize(img,600)

# img2 = cv.imread(path+'/'+str(taulells[random_num]))
# img2 = resize(img2,600)


hlines = hlines(img2)
vlines = vlines(img2)

img2 = dibuixar_linies(img2, hlines)
img2 = dibuixar_linies(img2, vlines)

punts = interseccions(hlines, vlines)
for i in range(len(punts)):
    for p in punts[i]:
        cv.circle(img2, (p[0],p[1]), 3, 255, -1)



# cv.imshow('1',img)
cv.imshow('2',img2)
cv.waitKey(0)
cv.destroyAllWindows()

# corners , shape = corners(img)
# lines = lines_corners(corners, shape)
# image3 = dibuixar_linies(img, lines)

# cv.imshow('hlines',image)
# cv.imshow('vlines',image2)
# cv.imshow('corners',image3)

# cv.waitKey(0)
# cv.destroyAllWindows()

# # print(punts)

def retallar_peces(img,punts): #función que recorta las piezas de un tablero dadas las coordenadas de las intersecciones
    peces = [[],[],[],[],[],[],[],[]]
    for i in range(len(punts)-1):#linies
        for j in range(len(punts[i])-1):#columnes
            u = 0 #up
            d = 0 #down
            r = 0 #right
            l = 0 #left
            fila_i = 0
            fila_f = 0
            columna_i = 0
            columna_f = 0
            if i < 3:
                if i == 0:
                    u = 1/2
                elif i == 1:
                    u = 1/3
                elif i == 2:
                    u = 1/4
            elif i > 4:
                if i == 5:
                    d = 1/4
                elif i == 6:
                    d = 1/3
                elif i == 7:
                    d = 1/2
            
            if j < 3:
                if j == 0:
                    l = 1/2
                elif j == 1:
                    l = 1/3
                elif j == 2:
                    l = 1/4
            elif j > 4:
                if j == 5:
                    r = 1/4
                elif j == 6:
                    r = 1/3
                elif j == 7:
                    r = 1/2
            
            if punts[i][j][1] <= punts[i][j+1][1]:
                fila_i = punts[i][j][1]
            else:
                fila_i = punts[i][j+1][1]
            
            if punts[i+1][j][1] >= punts[i+1][j+1][1]:
                fila_f = punts[i+1][j][1]
            else:
                fila_f = punts[i+1][j+1][1]
            
            if punts[i][j][0] <= punts[i+1][j][0]:
                columna_i = punts[i][j][0]
            else:
                columna_i = punts[i+1][j][0]
            
            if punts[i][j+1][0] >= punts[i+1][j+1][0]:
                columna_f = punts[i][j+1][0]
            else:
                columna_f = punts[i+1][j+1][0]
                           
            ample = columna_f - columna_i
            alt = fila_f - fila_i
            
            if fila_i-alt*u <= 0:
                fila_i = 1
            else:
                fila_i = fila_i-alt*u
                
            if fila_f+alt*d > img.shape[0]:
                fila_f = img.shape[0]
            else:
                fila_f = fila_f+alt*d
            
            if columna_i-ample*l <= 0:
                columna_i = 1
            else:
                columna_i = columna_i-ample*l
            
            if columna_f+ample*r > img.shape[1]:
                columna_f = img.shape[1]
            else:
                columna_f = columna_f+ample*r
            
            peça = img[round(fila_i):round(fila_f), round(columna_i):round(columna_f)]
            # peça = resize(peça,150)
            peça = cv.resize(peça,(128,128))
            peces[i].append(peça)
    
    return peces


# peces = retallar_peces(img, punts)

# for i in range(len(peces)):
#     for j in range(len(peces[i])):
#         cv.imshow('',peces[i][j])
#         cv.waitKey(0)
#         cv.destroyAllWindows()
        

        
                


