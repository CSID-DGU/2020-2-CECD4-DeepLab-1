from tkinter import *
from tkinter import filedialog
import tkinter.ttk as ttk
from PIL import ImageTk,Image
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os

root=Tk()
root.title("void_detection")
root.geometry("600x750+200+10")
root.resizable(width = False, height = True)
Image_names = []
myList = []
void_pixel = []


label_frame = Frame(root,width=450, height = 450, bg="white")
label_frame.pack_propagate(0)

my_label=Label(label_frame, bg = "white", width = 450, height = 450)
image_name = Label(root, text = "no image", font = ('arial',20))

label_frame.pack(pady = 20)
my_label.place(x = 0, y = 0)
image_name.pack(side = "top", pady = 10)


image_number = 0
def preprocessing(image) :
    origin = image
    ret, image = cv2.threshold(image, 60, 255, cv2.THRESH_TOZERO)
    ret, image = cv2.threshold(image, 150, 255, cv2.THRESH_TRUNC)
    height = image.shape[0]
    width = image.shape[1]
    count = -1
    while count != 0 :
        count = 0
        for i in range(2, height - 2) :
            for j in range(2, width - 2) :
                if(image[i, j] != 150 and image[i, j] != 0) :
                    if image[i + 1, j] == 0 or image[i, j + 1] == 0 or image[i - 1, j] == 0 or image[i, j - 1] == 0 or image[i + 1, j + 1] == 0 or image[i + 1, j - 1] == 0 or image[i - 1, j + 1] == 0 or image[i - 1, j - 1] == 0 :
                        image[i, j] = 0
                        count += 1
    
    for i in range(1, height - 1) :
        for j in range(j, width -1) :
            if image[i, j] > 0 :
                image[i, j] = 255
    kernel = np.ones((15, 15), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.add(origin, image)

    kernel = np.ones((21, 21), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.subtract(image, opening)
    for i in range(height) :
        for j in range(width) :
            pixel = image[i, j]
            if pixel * 5 <= 255 :
                image[i, j] = pixel * 5
            else :
                image[i, j] = 255
    image = cv2.add(image, 10)
    return image


def void_detection(image) :
    global void_pixel
    sum_void = 0

    image_pro = preprocessing(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    max_sigma = 10 # default: 30
    blobs_log = blob_log(image_pro, max_sigma=max_sigma, num_sigma=10, threshold=0.07)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)


    dx = [-1, 0, 1, 0, -1, -1, 1, 1]
    dy = [0, 1, 0, -1, -1, 1, -1, 1]


    for blob in blobs_log:
        y, x, r = blob


        if r < 1:
            small_radius += 1
            continue
            
        r = int(r)
        y = int(y)
        x = int(x)

        maxValCounts = []

        if r+4 < y-r and y+r < image_pro.shape[0]-r-4 and r+4 < x-r and x+r < image_pro.shape[1]-r-4:

            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.double)
            cv2.circle(mask, (x, y), r, (1.0), -1)
            mask = mask[y-2*r-4:y+2*r+4, x-2*r-4:x+2*r+4]

            ring1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.double)
            cv2.circle(ring1, (x, y), r+1, (1.0), 1)
            ring1 = ring1[y-2*r-4:y+2*r+4, x-2*r-4:x+2*r+4]

            ring2 = np.zeros((image.shape[0], image.shape[1]), dtype=np.double)
            cv2.circle(ring2, (x, y), r+2, (1.0), 1)
            ring2 = ring2[y-2*r-4:y+2*r+4, x-2*r-4:x+2*r+4]

            ring3 = np.zeros((image.shape[0], image.shape[1]), dtype=np.double)
            cv2.circle(ring3, (x, y), r+3, (1.0), 1)
            ring3 = ring3[y-2*r-4:y+2*r+4, x-2*r-4:x+2*r+4]

            ring4 = np.zeros((image.shape[0], image.shape[1]), dtype=np.double)
            cv2.circle(ring4, (x, y), r+3, (1.0), 1)
            ring4 = ring4[y-2*r-4:y+2*r+4, x-2*r-4:x+2*r+4]

            out = image_pro[y-2*r-4:y+2*r+4, x-2*r-4:x+2*r+4]
            
            inner = out[np.where(mask == 1.0)]
            outer1 = out[np.where(ring1 == 1.0)]
            outer2 = out[np.where(ring2 == 1.0)]
            outer3 = out[np.where(ring3 == 1.0)]
            outer4 = out[np.where(ring4 == 1.0)]

            innerAvg = sum(inner) / len(inner)
            avg1 = sum(outer1) / len(outer1)
            avg2 = sum(outer2) / len(outer2)
            avg3 = sum(outer3) / len(outer3)
            avg4 = sum(outer4) / len(outer4)
            threshold = innerAvg * 3 / 7
            cnt = 0
            if avg1 < threshold :
                cnt = cnt + 1
            if avg2 < threshold :
                cnt = cnt + 1
            if avg3 < threshold :
                cnt = cnt + 1
            if avg4 < threshold :
                cnt = cnt + 1
            
            min_avg = min(avg1, avg2, avg3)

            if (innerAvg - min_avg > 60 and innerAvg < 100) or (innerAvg >= 100 and cnt >= 1):
                image = cv2.circle(image, (x, y), r, (0, 0, 255), 2)
                sum_void += ((r**2) * 3.14)
    void_pixel.append(sum_void)
    return image

def resize(image) :
    height = image.shape[0]
    width = image.shape[1]
    if height > width :
        height = int(height *(450/height))
        width = int(width *(450/height))
    else :
        height = int(height *(450/width))
        width = int(width *(450/width))
    image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)
    return image
def choose_derectory() :
    global my_label
    global myList
    global info
    global info2
    global root
    global image_number
    global Image_names
    global image_name
    global button_back
    global void_pixel

    void_pixel.clear()
    myList.clear()
    Image_names.clear()
    image_number = 0
    button_back.config(state = DISABLED)

    root.directory = filedialog.askdirectory()
    directory = root.directory + "/"
    Image_names = os.listdir(directory)
    for i in range(len(Image_names)):
        origin_image = cv2.imread(directory+Image_names[i])
        image = void_detection(origin_image)
        image = resize(image)
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_coverted)
        resize_image = ImageTk.PhotoImage(image)
        myList.append(resize_image)
        void_pixel[i] = (void_pixel[i] / (origin_image.shape[0] * origin_image.shape[1])) * 100
    info.config(text = "void rate : " + str(round(void_pixel[image_number] ,2)))
    if void_pixel[image_number] > 2.0 :
        info2.config(text = "불량/정상 : 불량")
    else :
        info2.config(text = "불량/정상 : 정상")
    my_label.config(text = "", image = myList[0])
    image_name.config(text = Image_names[0])
    

def forward():
    global my_label
    global button_forward
    global button_back
    global info
    global info2
    global void_pixel
    global image_number
    global Image_names
    global image_name

    image_number += 1
    
    if image_number != 0 :
        button_back.config(state = NORMAL)
    if image_number==len(myList) - 1:
        button_forward.config(state = DISABLED)

    my_label.config(image=myList[image_number])
    info.config(text = "void rate : " + str(round(void_pixel[image_number] ,2)))
    if void_pixel[image_number] > 2.0 :
        info2.config(text = "불량/정상 : 불량")
    else :
        info2.config(text = "불량/정상 : 정상")
    image_name.config(text = Image_names[image_number])
    print(image_number)

def back():
    global my_label
    global image_number
    global button_forward
    global button_back
    global info
    global info2
    global void_pixel
    global Image_names
    global image_name
    
    image_number -= 1
    my_label.config(image=myList[image_number])
    image_name.config(text = Image_names[image_number])
    info.config(text = "void rate : " + str(round(void_pixel[image_number] ,2)))
    if void_pixel[image_number] > 2.0 :
        info2.config(text = "불량/정상 : 불량")
    else :
        info2.config(text = "불량/정상 : 정상")

    if image_number==0:
        button_back.config(state = DISABLED)

    if image_number != len(myList) :
        button_forward.config(state = NORMAL)
    print(image_number)
    

dir_button = Button(root, width = 20, text = "Select Directory", command = lambda : choose_derectory())
button_back=Button(root, width = 10, text="<<", command = lambda:back(), state=DISABLED)
button_forward=Button(root,width = 10, text=">>", command = lambda:forward())
info = Label(root, text = "void rate : ")
info2 = Label(root, text = "불량/정상 : ")

dir_button.pack(side = "top")
info.pack()
info2.pack()
button_back.pack(side = "left", anchor = "sw", padx = 80, pady = 20)
button_forward.pack(side = "right", anchor = "se", padx = 80, pady = 20)

root.mainloop()