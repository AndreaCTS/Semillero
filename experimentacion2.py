import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from scipy.stats import stats

imagenes = []
for i in range(2,7):
    enlace = "Photos/mdb001 ("+str(i)+").png"
    imagenes.append(cv.imread(enlace))
    imagenes[(i-2)] = cv.cvtColor(imagenes[(i-2)],cv.COLOR_RGB2GRAY)


#Comenando para poner en comentario lineas de código
#  (primero seleccionarlas): ctrl + }


#Imagen en escala de grises es igual a la original

#cv.imshow('Breast GREY',grey)
#cv.imshow('Breast',img)

#Para redimensionar la imagen
def rescaleFrame(frame, scale=1.75):
    #[1] porque se refiere al ancho de la imagen
    width = int(frame.shape[1] * scale)
    #[1] porque se refiere a la altura de la imagen
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(frame,dimensions, interpolation = cv.INTER_AREA)




#Para probar la función de redimensionar
#img_rescale = rescaleFrame(img1)

#imagen redimensionada
#cv.imshow('Breast resize',img_rescale)



#Dibujar un círculo en la imagen
#prueba en caso de ser necesario
#Dos formas diferentes de dibujar el ciírculo

# cv.circle(img,(img.shape[1]//2, img.shape[0]//2),40,(0,255,0),thickness=2)
# cv.circle(img,(150,250),40,(0,255,0),thickness=2)

#Muestro a img que es la imagen con el círculo 
#cv.imshow('Breast circle',img)

#cv.imshow('cat',img2)


#Bordes de la imagen
cv.imshow('Breast',imagenes[0])
canny = cv.Canny(imagenes[0],0,255)
cv.imshow('Breast edges canny',canny)

#TRESHOLD coloca la imagen en color blanco y negro
ret,thresh = cv.threshold(imagenes[0],125,175, cv.THRESH_BINARY)
cv.imshow("THRESH ",thresh)


contours, hierarchies = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contuors found: ')
'''
#Contours
'''


cv.waitKey(0)
