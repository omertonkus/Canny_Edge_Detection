from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import cv2

# resmin okunması
img = cv2.imread('food.jpeg')

# resmi gri yapma
img_gri = np.dot(img[...,:3], [0.299, 0.587, 0.114])

# Gri görüntüyü bulanıklaştırır, gürültülü olanlar göz ardı edilir.
img_gri_bulanik = ndimage.gaussian_filter(img_gri, sigma=1.4)

# Konvolusyon uygulanarak Sobel Filtresi
def SobelFilter(img, yon):
    if (yon == 'x'):
        Gx = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
        sonuc = ndimage.convolve(img, Gx)

    if (yon == 'y'):
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])
        sonuc = ndimage.convolve(img, Gy)

    return sonuc

# Pikselleri normalleştirme ( piksellerin değerini 1 den küçük hale getirmek yani değerleri daha dar bir alana indirgemek)
def Normalize(img):
    img = img/np.max(img)
    return img

# Sobel Filtresini x yönünde uygulama ( x yönündeki gradyeni bulma)
gx = SobelFilter(img_gri_bulanik, 'x')
gx = Normalize(gx)

# Sobel Filtresini y yönünde uygulama ( y yönündeki gradyeni bulma)
gy = SobelFilter(img_gri_bulanik, 'y')
gy = Normalize(gy)

# Gradyanların büyüklüğünü hesaplama
Mag = np.hypot(gx,gy) # karelerini alıp toplayıp karekökünü aldık
Mag = Normalize(Mag)

# Gradyanların yönünü hesaplama
Gradient = np.degrees(np.arctan2(gy,gx))

# Sahte kenarların maksimum olmayan bastırması
def NonMaxSup(Gmag, Grad):
    NMS = np.zeros(Gmag.shape)
    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if((Grad[i,j] >= -22.5 and Grad[i,j] <= 22.5) or (Grad[i,j] <= -157.5 and Grad[i,j] >= 157.5)):
                if((Gmag[i,j] > Gmag[i,j+1]) and (Gmag[i,j] > Gmag[i,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 22.5 and Grad[i,j] <= 67.5) or (Grad[i,j] <= -112.5 and Grad[i,j] >= -157.5)):
                if((Gmag[i,j] > Gmag[i+1,j+1]) and (Gmag[i,j] > Gmag[i-1,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 67.5 and Grad[i,j] <= 112.5) or (Grad[i,j] <= -67.5 and Grad[i,j] >= -112.5)):
                if((Gmag[i,j] > Gmag[i+1,j]) and (Gmag[i,j] > Gmag[i-1,j])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 112.5 and Grad[i,j] <= 157.5) or (Grad[i,j] <= -22.5 and Grad[i,j] >= -67.5)):
                if((Gmag[i,j] > Gmag[i+1,j-1]) and (Gmag[i,j] > Gmag[i-1,j+1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0

    return NMS

NMS = NonMaxSup(Mag, Gradient)
NMS = Normalize(NMS)

# Çift eşik Histerezis
def DoThreshHyst(img):
    highThresholdRatio = 0.2
    lowThresholdRatio = 0.15
    GSup = np.copy(img)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])
    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    x = 0.1
    oldx = 0

    while (oldx != x):
        oldx = x
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if (GSup[i, j] > highThreshold):
                    GSup[i, j] = 1
                elif (GSup[i, j] < lowThreshold):
                    GSup[i, j] = 0
                else:
                    if ((GSup[i - 1, j - 1] > highThreshold) or
                            (GSup[i - 1, j] > highThreshold) or
                            (GSup[i - 1, j + 1] > highThreshold) or
                            (GSup[i, j - 1] > highThreshold) or
                            (GSup[i, j + 1] > highThreshold) or
                            (GSup[i + 1, j - 1] > highThreshold) or
                            (GSup[i + 1, j] > highThreshold) or
                            (GSup[i + 1, j + 1] > highThreshold)):
                        GSup[i, j] = 1
        x = np.sum(GSup == 1)

    GSup = (GSup == 1) * GSup

    return GSup

son_resim = DoThreshHyst(NMS)
plt.imshow(son_resim, cmap = plt.get_cmap('gray'))
plt.show()