__author__ = '% Arisa Kitagishi and Barbara Cotter %'

import cv2
import glob
import time
import math
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter

'''images directory should by located inside your project main directory(Don't include it in the submission).'''
# Path to dataset images/
images = glob.glob('images/*.jpg')

class object_recognition():

    ''' If query_img = None, take the path to dataset by deault, else process single image and return the highes four of it.'''
    def matching(self, query_input = None, query=None):
        final_matches_list = []

        if query_input is not None:
            queue_label = query
            query_input = query_images_list
        else:
            queue_label = images
            query_input = template_images_list

        for i, inst in enumerate(queue_label):

            #Get the name of the image file without extension

            img = cv2.imread(inst, 0)
            color = ('b', 'g', 'r')
            for c, col in enumerate(color):
                hist = cv2.calcHist(img, [c], None, [256], [0, 256])
                plt.xlim([0, 256])
            for j, tmp in enumerate(images):
                template = cv2.imread(tmp, 0)
                '''TODO: Apply template Matching'''
                res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                template_matching = 1 if float(res[0])>0.9 else 0

                '''TODO: Apply color histogram. OpenCV function is more faster than (around 40X) than np.histogram(). So stick with OpenCV function.'''
                for c2, col2 in enumerate(color):
                       hist2 = cv2.calcHist(template, [c2], None, [256], [0, 256])
                       plt.xlim([0, 256])

                res = cv2.compareHist(hist, hist2, cv2.HISTCMP_CORREL)
                color_histogram = res

                '''TODO: Apply SIFT, SURF, or ORB detection'''
                # if python 2, and you want to use SIFT, pip uninstall opencv-python and pip uninstall opencv-contrib-python
                # and then  pip install opencv-python==3.4.2.16 and pip install opencv-contrib-python==3.4.2.16
                sift = cv2.xfeatures2d.SIFT_create()

                # find the keypoints and descriptors with SIFT
                kp1, des1 = sift.detectAndCompute(img, None)
                kp2, des2 = sift.detectAndCompute(template, None)

                # BFMatcher with default params
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)

                print 'printing sorted for image:' + str(i) + ' and template:' + str(j)
                # Apply ratio test
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])
                        print m.distance



                '''TODO: If you are a group of 2, make sure to create your custom algorithm and added to the list after detection'''
                orb = cv2.ORB_create(nfeatures=20)

                kp, des = orb.detectAndCompute(img, None)
                kp2, des2 = orb.detectAndCompute(template, None)

                # Brute Force Matching
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des, des2)
                sorted_matches = sorted(matches, key=lambda x: x.distance)


                average = 0
                total = 0
                for k in range(len(sorted_matches)):
                    average += sorted_matches[k].distance
                    total += 1
                average /= total
                #print sorted_matches[0].distance
                detection = sorted_matches[0].distance

                final_matches_list.append([inst.partition('\\')[2].split('.')[0], template_matching, color_histogram, detection])

        return final_matches_list


    '''Return a dictionary with the score values and the top 4 dictionary'''
    def scoring(self, match_result_list):

        sorted_top4 = []
        temp_sort_list = []
        sorted_scored_list = []
        temp_index = 0
        score = 0
        result = [0.0,0.0,0.0,0.0]
        result_list = []
        score_list = [0,0,0]
        for i in match_result_list:
            if i[0] == match_result_list[temp_index][0]:
                temp_sort_list.append(i)
            else:
                '''sort the newly gotten scores for Template Matching'''
                for j in range(3):
                    sorted_list = sorted(temp_sort_list,key = lambda x: x[j+1], reverse=True)
                    '''get the top4'''
                    for k in range(4):
                        if sorted_list[k][j+1] > 0.9:
                          score +=1
                        result[k] = sorted_list[k][j+1]
                    result_list.append(result)
                    score_list[j] = score
                    result = [0.0,0.0,0.0,0.0]

                print '***for ' + str(match_result_list[temp_index][0]) + '***'
                print '\t all scores sorted'
                print sorted_list
                print '\t topq 4 scores'
                for l in range(4):
                    sorted_top4.append([temp_sort_list[l][0], result_list[0][l], result_list[1][l], result_list[2][l]])
                print sorted_top4

                sorted_scored_list.append([match_result_list[temp_index][0], score_list[0], score_list[1], score_list[2]])
                '''clear for next image'''
                temp_sort_list = []
                sorted_top4 = []
                score=0
                '''update temp_index to keep track of change in images'''
                temp_sort_list.append(i)
                temp_index = match_result_list.index(i)

        return match_result_list, sorted_scored_list


    def eval_cal(self, scoring_result):
        pass
        mean_tm = 0.0
        mean_ch = 0.0
        mean_dt = 0.0
        total = 0
        '''Calculate the mean'''
        for i in scoring_result:
            mean_tm += scoring_result[total][1]
            mean_ch += scoring_result[total][2]
            mean_dt += scoring_result[total][3]
            total += 1

        mean_tm /= total
        mean_ch /= total
        mean_dt /= total
        '''Uncomment the line below after'''
        return [["Template_match_mean", mean_tm], ["color_histogram_mean", mean_ch], ["sift_detection_mean", mean_dt]]


if __name__ == '__main__':

  query_images_list = []
  template_images_list = []
  scores = np.zeros((20, 20), dtype=int)

  '''Specify a single image or multiple images name in the query. See example below'''
  #query = ['images/ukbench00021.jpg', 'images/ukbench01492.jpg']

  '''If None, then iterate through all the 20 images'''
  query = None

  try:
      obj_rec = object_recognition()

      '''Run on query_img if argument passed, otherwise run on dataset'''
      if query != None:

          for query_img in query:
              query_images_list.append(cv2.imread(query_img, 0))

      for i in range(len(images)):
          template_images_list.append(cv2.imread(images[i], 0))

      matching_results = obj_rec.matching(query)
      scoring_results, sorted_scored_top4 = obj_rec.scoring(matching_results)
      evaluation_results = obj_rec.eval_cal(scoring_results)

      print(dict(evaluation_results))

  except KeyboardInterrupt:
    print("Shutting down")
