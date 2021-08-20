import urllib.request as req
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def Img(img, bg=False):
    try:
        if bg == True:
            image = cv.imread(img, -1)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            c_find, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv.drawContours(mask, c_find, -1, color=(255, 255, 255), thickness=cv.FILLED)
            fg = cv.bitwise_and(gray, gray, mask=mask)
            bg = cv.bitwise_not(mask)
            image = cv.bitwise_or(fg, bg)
        else:
            image = cv.imread(img, 0)

        return image
    except E as Exception:
        print(E)

def urlImg(url, bg=False):
    try:
        resp = req.urlopen(url)
        img_array = np.asarray(bytearray(resp.read()), dtype="uint8")
        
        if bg == True:
            image = cv.imdecode(img_array, -1)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            c_find, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv.drawContours(mask, c_find, -1, color=(255, 255, 255), thickness=cv.FILLED)
            fg = cv.bitwise_and(gray, gray, mask=mask)
            bg = cv.bitwise_not(mask)
            image = cv.bitwise_or(fg, bg)
        else:
            image = cv.imdecode(img_array, 0)

        return image
    except Exception as E:
        print(E)

def SCALE(template, percent):
    scalepercent = percent
    wscaled = int(template.shape[1] * scalepercent / 100)
    hscaled = int(template.shape[0] * scalepercent / 100)
    return (wscaled, hscaled)

models = [
    {"image": r"C:\Users\joshu\Desktop\projects\detection\1\imgs\model1.jfif"},
    {"image": r"C:\Users\joshu\Desktop\projects\detection\1\imgs\model2.jfif"},
]

samples = [
    {"image": r"C:\Users\joshu\Desktop\projects\detection\1\imgs\sample1.png", "bg": False, "scale": 100, "threshold": .8},
    {"image": r"C:\Users\joshu\Desktop\projects\detection\1\imgs\sample2.png", "bg": False, "scale": 22, "threshold": .22},
    {"image": 'https://tr.rbxcdn.com/bb29fd64ac1bc170e4cbb9a836eddb34/150/150/AvatarHeadshot/Png', "bg": True, "scale": 60, "threshold": .6}
]

methods = ['cv.TM_CCOEFF_NORMED']

#preinit img instead of loading it in the for loop?

for m in models:
    try:
        try:
            model_template = Img(m["image"]).copy()
            model = cv.resize(model_template, SCALE(model_template, 100), interpolation=cv.INTER_AREA)
        except:
            model_template = urlImg(m["image"]).copy()
            model = cv.resize(model_template, SCALE(model_template, 100), interpolation=cv.INTER_AREA)

        for method in methods:
            for s in samples:
                try:
                    try:
                        sample_img = Img(s["image"], s["bg"]).copy()
                        sample = cv.resize(sample_img, SCALE(sample_img, s["scale"]), interpolation=cv.INTER_AREA)
                    except:
                        sample_img = urlImg(s["image"], s["bg"]).copy()
                        sample = cv.resize(sample_img, SCALE(sample_img, s["scale"]), interpolation=cv.INTER_AREA)

                    w, h = sample.shape[::-1]

                    mdl = model.copy()
                    mtd = eval(method)

                    res = cv.matchTemplate(mdl, sample, mtd)
                    threshold = s["threshold"]
                    loc = np.where(res >= threshold)
                    for pt in zip(*loc[::-1]):
                        cv.rectangle(mdl, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

                    plt.subplot(121), plt.imshow(sample, cmap='gray')
                    plt.title('IMG'), plt.xticks([]), plt.yticks([])
                    # plt.subplot(121),plt.imshow(res,cmap = 'gray')
                    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                    plt.subplot(122), plt.imshow(mdl, cmap='gray')
                    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                    plt.suptitle(method)
                    plt.show()
                except Exception as E:
                    print(E)
    except Exception as E:
        print(E)