__author__ = '% Arisa Kitagishi and Barbara Cotter %'

# Instructions:
# Please have all images under the 'images' folder
# So the path is 'images/'
# To be able to run SIFT, pip uninstall opencv-python and pip uninstall opencv-contrib-python.
# Then, pip install opencv-python==3.4.2.16 and pip install opencv-contrib-python==3.4.2.16
# Thank you!

import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt

# Path to dataset images/
images = glob.glob('images/*.jpg')

class object_recognition():

    def matching(self):
        final_matches_list = []
        # Iterate through all the images to apply feature matching
        for i, inst in enumerate(images):
            # Read the images one by one which are stored in queue_label using OpenCV
            img = cv2.imread(inst, 0)

            # For each templates, iterate them to compare against the current image
            for j, tmp in enumerate(images):
                # Read the template using the OpenCV
                template = cv2.imread(tmp, 0)
                # Apply template matching using OpenCV
                # Use TM_CCOEFF_NORMED as this performed the best according to the last OpenCV assignment
                res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                # Store the raw result of the template matching to pass to next method later
                template_matching = float(res)

                # Apply Color Histogram Based Matching
                # Initialize color list to iterate easier through the colors
                color = ('b', 'g', 'r')
                # For each color: b, g, r calculate the histogram for the current image
                for c, col in enumerate(color):
                    hist = cv2.calcHist(img, [c], None, [256], [0, 256])
                    plt.xlim([0, 256])
                # For each color: b, g, r calculate the histogram for the template
                for c2, col2 in enumerate(color):
                    hist2 = cv2.calcHist(template, [c2], None, [256], [0, 256])
                    plt.xlim([0, 256])
                # Store the raw result of the template matching to pass to next method later
                res = cv2.compareHist(hist, hist2, cv2.HISTCMP_CORREL)
                color_histogram = res

                # Apply SIFT with Ratio Test using OpenCV
                sift = cv2.xfeatures2d.SIFT_create()
                # Find the keypoints and descriptors with SIFT
                kp1, des1 = sift.detectAndCompute(img, None)
                kp2, des2 = sift.detectAndCompute(template, None)
                # BFMatcher with default params
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                # Apply the Ratio Test
                valid_distance = 0
                good = []
                # For each distance recognized as good by ratio test,
                # Count them as valid_distance because if the template has more, then it has more validity
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])
                        valid_distance += 1
                # Store the amount of valid distances and use that to find top 4 matching images
                detection = valid_distance

                # Apply ORB with Ratio Test using OpenCV
                orb = cv2.ORB_create()
                # Find the keypoints and descriptors with SIFT
                kp, des = orb.detectAndCompute(img, None)
                kp2, des2 = orb.detectAndCompute(template, None)
                # BFMatcher with default params
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des, des2, k=2)
                # Apply ratio test
                valid_distance = 0
                good = []
                # For each distance recognized as good by ratio test,
                # Count them as valid_distance because if the template has more, then it has more validity
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])
                        valid_distance += 1
                # Store the amount of valid distances and use that to find top 4 matching images
                detection2 = valid_distance

                # Store the name of image, results from all four methods, and the name of the template and return this
                final_matches_list.append(
                    [inst.partition('\\')[2].split('.')[0], template_matching, color_histogram, detection, detection2,
                     tmp.partition('\\')[2].split('.')[0]])
        return final_matches_list

    def scoring(self, match_result_list):
        temp_index = 0  # This stores the previous index before the image changes
        temp_sort_list = []
        sorted_scored_list = []
        score = 0
        result = []
        score_list = [0, 0, 0, 0]
        methods = ["Template Matching", "Color Histogram", "SIFT", "ORB"]
        # Iterate through all the results found in match_result_list which contains
        # [image name, template matching result, color histogram result, SIFT result, ORB result, template name]
        for i in match_result_list:
            # If the image name stayed the same, append it to the temp_sort_list to sort later
            if i[0] == match_result_list[temp_index][0]:
                temp_sort_list.append(i)
            # If the image name changed, then sort and find the scores and top 4 using the list obtained earlier
            else:
                print '***Top 4 images with ' + str(match_result_list[temp_index][0]) + '***'
                # Go through all four methods
                for j in range(4):
                    # Sort the list based on the score of the current method
                    sorted_list = sorted(temp_sort_list, key=lambda x: x[j + 1], reverse=True)
                    print 'for method: ' + str(methods[j])
                    # Iterate four times as we need four scores
                    for k in range(4):
                        # Find the top 4 by using the sorted list
                        result.append([sorted_list[k][5], sorted_list[k][j + 1]])
                        # By comparing the name of the images, see if the method found a match
                        if abs(int(sorted_list[k][5].partition('ukbench0')[2].split('.jpg')[0]) -
                               int(match_result_list[temp_index][0].partition('ukbench0')[2].split('.jpg')[0])) < 3:
                            # Increment score if it is a match
                            score += 1
                    # Print the method's top 4 results with the associated template name
                    print result
                    # Store the obtained score for this method into score_list
                    score_list[j] = score
                    # Empty result and score for next method
                    result = []
                    score = 0
                # Store the scores obtained in order of template matching, color histogram, SIFT, and ORB
                sorted_scored_list.append([score_list[0], score_list[1], score_list[2], score_list[3]])
                # Print all the scores for this image
                print 'Scores for all the methods: Template Matching, Color Histogram, SIFT, ORB respectively'
                print sorted_scored_list
                # Store the scores into categories for mean and standard deviation
                templateAll.append(float(score_list[0]))
                histAll.append(float(score_list[1]))
                siftAll.append(float(score_list[2]))
                orbAll.append(float(score_list[3]))
                # Clear the lists and score for next image
                temp_sort_list = []
                sorted_scored_list = []
                score = 0
                # Add the current image into list and update temp_index
                temp_sort_list.append(i)
                temp_index = match_result_list.index(i)
        # Do the procedure one more time for the last image
        print '***Top 4 images with ' + str(match_result_list[temp_index][0]) + '***'
        print '(name of image, result of match)'
        # Go through all four methods
        for j in range(4):
            # Sort the list based on the score of the current method
            sorted_list = sorted(temp_sort_list, key=lambda x: x[j + 1], reverse=True)
            '''get the top4'''
            print 'for method: ' + str(methods[j])
            # Iterate four times as we need four scores
            for k in range(4):
                # Find the top 4 by using the sorted list
                result.append([sorted_list[k][5], sorted_list[k][j + 1]])
                # By comparing the name of the images, see if the method found a match
                if abs(int(sorted_list[k][5].partition('ukbench0')[2].split('.jpg')[0]) -
                       int(match_result_list[temp_index][0].partition('ukbench0')[2].split('.jpg')[0])) < 3:
                    # Increment score if it is a match
                    score += 1
            # Print the method's top 4 results with the associated template name
            print result
            # Store the obtained score for this method into score_list
            score_list[j] = score
            # Empty result and score for next method
            result = []
            score = 0

        # Store the scores obtained in order of template matching, color histogram, SIFT, and ORB
        sorted_scored_list.append([score_list[0], score_list[1], score_list[2], score_list[3]])
        # Print all the scores for this image
        print 'Scores for all the methods: Template Matching, Color Histogram, SIFT, ORB respectively'
        print sorted_scored_list
        # Store the scores into categories for mean and standard deviation
        templateAll.append(float(score_list[0]))
        histAll.append(float(score_list[1]))
        siftAll.append(float(score_list[2]))
        orbAll.append(float(score_list[3]))

        return match_result_list, sorted_scored_list

    def eval_cal(self, scoring_result):
        pass
        mean_tm = 0.0
        mean_ch = 0.0
        mean_dt = 0.0
        mean_dt2 = 0.0
        total = 0
        # Sum all the results found in scoring_result
        for i in scoring_result:
            mean_tm += scoring_result[total][1]
            mean_ch += scoring_result[total][2]
            mean_dt += scoring_result[total][3]
            mean_dt2 += scoring_result[total][4]
            # Use total to find the total number of scores
            total += 1
        # Find the mean
        mean_tm /= total
        mean_ch /= total
        mean_dt /= total
        mean_dt2 /= total
        return [["Template_match_mean", mean_tm], ["color_histogram_mean", mean_ch], ["sift_detection_mean", mean_dt],
                ["orb_detection_mean", mean_dt2]]


if __name__ == '__main__':

    query_images_list = []
    template_images_list = []
    templateAll = []
    histAll = []
    siftAll = []
    orbAll = []
    addAll = [0, 0, 0, 0]
    finalMean = [0, 0, 0, 0]
    standard_dev = [20]

    try:
        obj_rec = object_recognition()
        # For all the images stored, store them as template images
        for i in range(len(images)):
            template_images_list.append(cv2.imread(images[i], 0))
        # Call methods
        matching_results = obj_rec.matching()
        scoring_results, sorted_scored_top4 = obj_rec.scoring(matching_results)
        evaluation_results = obj_rec.eval_cal(scoring_results)

        # Print all scores for template matching
        print 'All Scores for Template Matching:'
        print templateAll
        # Print all scores for color histogram
        print 'All Scores for Color Histogram:'
        print histAll
        # Print all scores for SIFT
        print 'All Scores for SIFT:'
        print siftAll
        # Print all scores for ORB
        print 'All Scores for ORB:'
        print orbAll

        # Turn the array of scores into numpy array
        t = np.array(templateAll)
        h = np.array(histAll)
        s = np.array(siftAll)
        o = np.array(orbAll)
        # Perform standard deviation function on those numpy arrays
        stdT = t.std()
        stdH = h.std()
        stdS = s.std()
        stdO = o.std()
        # Combine all scores for template matching
        sum = 0
        for score_sum in templateAll:
            sum += score_sum
        addAll[0] = sum
        # Find the average
        finalMean[0] = addAll[0] / len(templateAll)
        # Combine all scores for color histogram
        sum = 0
        for score_sum in histAll:
            sum += score_sum
        addAll[1] = sum
        # Find the average
        finalMean[1] = addAll[1] / len(histAll)
        # Combine all scores for SIFT
        sum = 0
        for score_sum in siftAll:
            sum += score_sum
        addAll[2] = sum
        # Find the average
        finalMean[2] = addAll[2] / len(siftAll)
        # Combine all scores for ORB
        sum = 0
        for score_sum in orbAll:
            sum += score_sum
        addAll[3] = sum
        # Find the average
        finalMean[3] = addAll[3] / len(orbAll)
        # Print the means of raw results for each methods
        print(evaluation_results)
        # Print the means of the scores for each methods
        print finalMean
        # Print the standard deviation for template matching
        print 'Standard Deviation for Template Matching:'
        print stdT
        # Print the standard deviation for color histogram
        print 'Standard Deviation for Histogram:'
        print stdH
        # Print the standard deviation for SIFT
        print 'Standard Deviation for SIFT:'
        print stdS
        # Print the standard deviation for ORB
        print 'Standard Deviation for ORB:'
        print stdO

    except KeyboardInterrupt:
        print("Shutting down")
