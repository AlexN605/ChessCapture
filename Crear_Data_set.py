from detectar_taulell import  *
import cv2 as cv
import numpy as np
import math
import sympy as sp
import os
import random
import pandas as pd
import operator

path = 'C:/Users/alexn/Desktop/TR/Data_set'
carpetes = os.listdir(path)
tipus_peces = [['pb','tb','cb','ab','kb','qb','e'],['pn','tn','cn','an','kn','qn']]

 #----------------Crear carpetas en Data_set---------------------------

if not 'Taulells' in carpetes:
    os.mkdir(path+'/Taulells')

if not 'txt' in carpetes:
    os.mkdir(path+'/txt')

if not 'Peces' in carpetes:
    os.mkdir(path+'/Peces')
    
for i in tipus_peces[0]:
    if not i in os.listdir(path+'/Peces'):
        os.mkdir(path+'/Peces'+'/'+i)
for i in tipus_peces[1]:
    if not i in os.listdir(path+'/Peces'):
        os.mkdir(path+'/Peces'+'/'+i)

taulells = os.listdir(path+'/Taulells')

#-------------------Crear txt-------------------------------------------------------

for i in range(len(taulells)):
     nom = taulells[i].split('.')[0]
     with open(path+'/txt2/'+nom+'.txt','w') as txt:
        pass
#-------------------------------------------------------------------------
#-------------Recortar y clasificar en carpetas ------------------------
for i in range(len(taulells)):
    nom = taulells[i].split('.')[0]
    taulell = cv.imread(path+'/Taulells/'+str(taulells[i]))
    taulell = resize(taulell,600)
    linesh = hlines(taulell)
    linesv = vlines(taulell)
    punts = interseccions(linesh,linesv)
    peces = retallar_peces(taulell,punts)
    distribucio = []
    print(distribucio)
    with open(path+'/txt/'+str(nom)+'.txt') as distribucio_txt:
        
        lines = distribucio_txt.read().split('\n')
        
        for line in lines:
            distribucio.append(line.split(' '))
        # print(distribucio)
    for i in range(len(peces)):
        for j in range(len(peces[i])):
            if distribucio[i][j] in tipus_peces[0] or distribucio[i][j] in tipus_peces[1]:
                random_num = random.randint(0,999999999)
                while random_num in os.listdir(path+'/Peces/'+distribucio[i][j]):
                    random_num = random.randint(0,999999999)
                cv.imwrite(path+'/Peces/'+distribucio[i][j]+'/'+nom+'_'+str(random_num)+'.jpg', peces[i][j])
            else:
                print('ERROR en etiquetacio en la foto {}'.format(nom))
#----------------------------Crear carpetas para dividir el dataset------
if not 'train' in os.listdir(path):
    os.mkdir(path+'/train')
    
if not 'test' in os.listdir(path):
    os.mkdir(path+'/test')
    
if not 'valid' in os.listdir(path):
    os.mkdir(path+'/valid')

for i in tipus_peces[0]:
    if not i in os.listdir(path+'/train'):
        os.mkdir(path+'/train'+'/'+i)
for i in tipus_peces[1]:
    if not i in os.listdir(path+'/train'):
        os.mkdir(path+'/train'+'/'+i)
        
for i in tipus_peces[0]:
    if not i in os.listdir(path+'/test'):
        os.mkdir(path+'/test'+'/'+i)
for i in tipus_peces[1]:
    if not i in os.listdir(path+'/test'):
        os.mkdir(path+'/test'+'/'+i)

for i in tipus_peces[0]:
    if not i in os.listdir(path+'/valid'):
        os.mkdir(path+'/valid'+'/'+i)
for i in tipus_peces[1]:
    if not i in os.listdir(path+'/valid'):
        os.mkdir(path+'/valid'+'/'+i)
        
carpetes_peces = os.listdir(path+'/Peces')

#-----------------repartir el dataset en test / train / validation------------------------------------
for i in carpetes_peces: 
    fotos = os.listdir(path+'/Peces/'+str(i))
    test = []
    train = []
    valid = []
    for j in range(int(len(fotos)//6.66)): #15% del total inicial
        foto = random.choice(fotos)
        test.append(foto)
        fotos.remove(foto)
    for j in range(int(len(fotos)//5.66)): #15% del total inicial
        foto = random.choice(fotos)
        valid.append(foto)
        fotos.remove(foto)
    for j in range(int(len(fotos))): #el 70% que queda a la llista
        train.append(fotos[j])
    for j in train:
        foto = cv.imread(path+'/Peces/'+str(i)+'/'+str(j))
        cv.imwrite(path+'/train/'+str(i)+'/'+str(j),foto)
    for j in test:
        foto = cv.imread(path+'/Peces/'+str(i)+'/'+str(j))
        cv.imwrite(path+'/test/'+str(i)+'/'+str(j),foto)
    for j in valid:
        foto = cv.imread(path+'/Peces/'+str(i)+'/'+str(j))
        cv.imwrite(path+'/valid/'+str(i)+'/'+str(j),foto)
    

        
    
    