# -*- coding: UTF-8 -*-
# Arisa Kitagishi and Barbara Cotter
# OpenCV Project

# Instructions for use:
# 1: Please place all 20 pictures in same directory as .py file
# 2: When prompted please paste the path
#    for the directory the picture are located
#    then press enter
# 3: If you would like to see every picture before you
#    start the program 1st type Y else type N

import cv2
import time
import numpy as np

from matplotlib import pyplot as plt
import glob

def main():
    list_of_pics = []
    list_of_pics_gray_scale = []
    list_of_pics_hist = []
    scores = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    print "Please copy and paste the path where pictures are located"
    direct = raw_input()
    directory = glob.glob(direct+"\*.jpg")
    for picfiles in directory:
        image = cv2.imread(picfiles)
        image_gray = cv2.imread(picfiles, 0)
        list_of_pics.append(image)
        list_of_pics_gray_scale.append(image_gray)

    j = 0
    while j == 0:
        print "If you would like to see the pictures before we start the program " \
              "please type Y for yes and N for no"
        in_put = raw_input()
        if in_put == 'Y':
            i = 0
            while i < 20:
                cv2.imshow("list_of_pics",np.array(list_of_pics[i]))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print "Image Shape:", np.array(list_of_pics[i]).shape
                i += 1

            i = 0
            while i < 20:
                cv2.imshow("list_of_pics_gray_scale", np.array(list_of_pics_gray_scale[i]))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print "Image Shape:", np.array(list_of_pics_gray_scale[i]).shape
                i += 1
            break
        elif in_put == 'N':
            break
        else:
            print "............................................________ "
            time.sleep(0.1)
            print "....................................,.-‘”...................``~., "
            time.sleep(0.1)
            print ".............................,.-”...................................“-., "
            time.sleep(0.1)
            print ".........................,/...............................................”:, "
            time.sleep(0.1)
            print ".....................,/......................................................\, "
            time.sleep(0.1)
            print".................../...........................................................,} "
            time.sleep(0.1)
            print "................./......................................................,:`''`..} "
            time.sleep(0.1)
            print ".............../...................................................,:”........./ "
            time.sleep(0.1)
            print "............../.....__.........................................:`.........../"
            time.sleep(0.1)
            print "............./__.(.....“~-,_..............................,:`........../ "
            time.sleep(0.1)
            print ".........../(_....”~,_........“~,_....................,:`........_/ "
            time.sleep(0.1)
            print "..........{.._`;_......”=,_.......“-,_.......,.-~-,},.~”;/....}"
            time.sleep(0.1)
            print "...........((.....`~_.......”=-._......“;,,./`..../”............../"
            time.sleep(0.1)
            print "...,,,___.\`~,......“~.,....................`.....}............../ "
            time.sleep(0.1)
            print "............(....`=-,,.......`........................(......;_,,-” "
            time.sleep(0.1)
            print "............/.`~,......`-...............................\....../\ "
            time.sleep(0.1)
            print ".............\`~.*-,.....................................|,./.....\,__ "
            time.sleep(0.1)
            print ",,_..........}.>-._\...................................|..............`=~-, "
            time.sleep(0.1)
            print ".....`=~-,_\_......`\,.................................\ ................."
            time.sleep(0.1)
            print "...................`=~-,,.\,...............................\ ....................."
            time.sleep(0.1)
            print "................................`:,,...........................`\..............__ ......."
            time.sleep(0.1)
            print ".....................................`=-,...................,%`>--==`` ................"
            time.sleep(0.1)
            print "........................................_\..........._,-%.......`\ ......................"
            time.sleep(0.1)
            print "...................................,<`.._|_,-&``................`\ "
            time.sleep(0.1)
            print"Really?!?!? Please try again\n"
            time.sleep(3)

    print "Image List Shape:", np.array(list_of_pics).shape
    print "Image List Gray Shape:", np.array(list_of_pics_gray_scale).shape
    method = 'cv2.TM_CCOEFF_NORMED'
    method = eval(method)

    print "                         Method 1                           "
    print "O(>V<)O                                              O(>V<)O"
    time.sleep(0.5)
    print "    _______ ___  __   __  ____  _       ___  _______ ___    "
    time.sleep(0.5)
    print "    __   __|    |  | |  ||    || |     / _ \ __   __|       "
    time.sleep(0.5)
    print "      | |  |--- |  | |  ||____|| |__  / /_\ \  | |  |---    "
    time.sleep(0.5)
    print "      | |  |___ |   |   ||     |____|/_/   \_\ | |  |___    "
    time.sleep(0.5)
    print " __   __    ___  _______  ____  _    _ _______  _      ____ "
    time.sleep(0.5)
    print "|  | |  |  / _ \ __   __ |  __|| |__| |__   __ | \  | |     "
    time.sleep(0.5)
    print "|  | |  | / /_\ \  | |   | |__ |  __  |__| |__ |  \ | | ---|"
    time.sleep(0.5)
    print "|   |   |/_/   \_\ | |   |____||_|  |_|_______ |   \| |____|"
    time.sleep(3)

    i = 0
    for query in list_of_pics_gray_scale:
        k = 0
        while k < 20:
            pic = list_of_pics_gray_scale[k]
            # Apply template Matching
            res = cv2.matchTemplate(pic, query, method)
            scores[i][k] = res
            if scores[i][k] >= .9 and scores[i][k] <= 1:
                print "Match of query image "+str(i+1)+" to image "+str(k+1)
            k += 1
        i += 1
    print ""
    print "As we can see using template matching does not"
    time.sleep(1)
    print "give good results because it has to almost be an"
    time.sleep(1)
    print "exact match for it to find a match"
    time.sleep(5)
    print ""

    print "                                       Method 2                                    "
    print "O(>V<)O                                                                     O(>V<)O"
    time.sleep(0.5)
    print " _    _  _______ _______ _______  ______  ____    ______   ___    __   __  _______ "
    time.sleep(0.5)
    print "| |__| | __   __ |       __   __ |  __  | |      | _____| / _ \  |  | |  | |       "
    time.sleep(0.5)
    print "|  __  | __| |__  ------|  | |   | |__| | | ---| |  \    / /_\ \ |  | |  |  ------|"
    time.sleep(0.5)
    print "|_|  |_| _______ _______|  | |   |______| |____| |    \ /_/   \_\|   |   | _______|"
    time.sleep(3)

    color = ('b', 'g', 'r')
    j = 0
    while j < 20:
        for i, col in enumerate(color):
            hist = cv2.calcHist([list_of_pics[j]], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        list_of_pics_hist.append(hist)
        plt.show()
        j += 1

    i = 0
    for query in list_of_pics_hist:
        k = 0
        while k < 20:
            pic = list_of_pics_hist[k]
            # Apply template Matching
            res = cv2.compareHist(query, pic, cv2.HISTCMP_CORREL)
            scores[i][k] = res
            if scores[i][k] >= .9 and scores[i][k] <= 1:
                print "Match of query image "+str(i+1)+" to image "+str(k+1)
            k += 1
        i += 1


    time.sleep(5)
    print ""
    print "             Method 3            "
    print "O(>V<)O                   O(>V<)O"
    time.sleep(0.5)
    print "_______  _______ _______ _______ "
    time.sleep(0.5)
    print "|        __   __ |       __   __ "
    time.sleep(0.5)
    print " ------| __| |__ |------   | |   "
    time.sleep(0.5)
    print "_______| _______ |         | |   "
    time.sleep(3)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(list_of_pics_gray_scale[0], None)
    img = cv2.drawKeypoints(list_of_pics_gray_scale[0], kp)
    cv2.imshow("pic", img)


if __name__ == "__main__":
    main()