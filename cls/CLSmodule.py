import os
import argparse
import os
import glob
import numpy as np
import cv2
from PIL import Image
from skimage.morphology import skeletonize
import mediapipe as mp

def extract_feature(line, image_height, image_width):
    image_size = np.array([image_height, image_width], dtype=np.float32)
    feature = np.append(np.min(line, axis=0)[:2]/image_size, np.max(line, axis=0)[:2]/image_size)
    feature *= 10
    N = 10
    print(line)
    step = len(line)//N
    print("Step :", step)
    for i in range(N):
        l = line[i*step:(i+1)*step]
        print("I :", l)
        mean_values = np.mean(l, axis=0)
        print("Mean_Values : ", mean_values)
        if mean_values.size >= 3:
            feature = np.append(feature, mean_values[2:])
            print("size 3 이상")
            print("Temp Feature : ", feature)
        else:
            print("else")
            feature = np.append(feature, np.zeros(3 - mean_values.size))

    print("Feature : ", feature)
    return feature

def group(img):
    # (1) build a graph
    # (1)-1 find all nodes
    count = np.zeros(img.shape)
    nodes = []

    for j in range(1, img.shape[0] - 1):
        for i in range(1, img.shape[1] - 1):
            if img[j, i] == 0: continue
            count[j, i] = np.count_nonzero(img[j-1:j+2, i-1:i+2]) - 1
            if count[j, i] == 1 or count[j, i] >= 3:
                nodes.append((j, i))

    # sort nodes to traverse from upper-left to lower-right
    nodes.sort(key = lambda x : x[0]+x[1])

    # (1)-2 save all connections
    graph = dict()
    for node in nodes:
        graph[node] = dict()

    not_visited = np.ones(img.shape)
    for node in nodes:
        y,x = node
        not_visited[y, x] = 0
        around = np.multiply(count[y-1:y+2, x-1:x+2], not_visited[y-1:y+2, x-1:x+2])
        next_pos = np.transpose(np.nonzero(around))

        if next_pos.shape[0] == 0: continue
        for dy,dx in next_pos:
            y,x = node
            next_y = y + dy - 1
            next_x = x + dx - 1
            if dx == 0 or (dy == 0 and dx == 1):
                dy,dx = 2-dy,2-dx
            temp_line = [[y,x,0,0], [next_y,next_x,dy-1,dx-1]]
            if count[next_y, next_x] == 1 or count[next_y, next_x] >= 3:
                not_visited[next_y, next_x] = 1
                graph[tuple(temp_line[0][:2])][tuple(temp_line[-1][:2])] = temp_line
                temp_line_rev = list(reversed(temp_line))
                graph[tuple(temp_line[-1][:2])][tuple(temp_line[0][:2])] = temp_line_rev
                continue

            while(True):
                y,x = temp_line[-1][:2]
                not_visited[y, x] = 0
                around = np.multiply(count[y-1:y+2, x-1:x+2], not_visited[y-1:y+2, x-1:x+2])
                next_pos = np.transpose(np.nonzero(around))
                if next_pos.shape[0] == 0: break

                # update line
                next_y = y + next_pos[0][0] - 1
                next_x = x + next_pos[0][1] - 1
                dy,dx = next_y-y,next_x-x
                if dx == -1 or (dy == -1 and dx == 0):
                    dy,dx = -dy,-dx
                temp_line.append([next_y, next_x, dy, dx])
                not_visited[next_y, next_x] = 0

                # check end condition
                if count[next_y, next_x] == 1 or count[next_y, next_x] >= 3:
                    #if len(temp_line) > 10:
                    graph[tuple(temp_line[0][:2])][tuple(temp_line[-1][:2])] = temp_line
                    temp_line_rev = list(reversed(temp_line))
                    graph[tuple(temp_line[-1][:2])][tuple(temp_line[0][:2])] = temp_line_rev
                    not_visited[next_y, next_x] = 1
                    break
        not_visited[node[0], node[1]] = 1

    lines_node = []
    visited_node = dict()
    finished_node = dict()
    for node in nodes:
        visited_node[node] = False
        finished_node[node] = False

    for node in nodes:
        if not finished_node[node]:
            temp = [node]
            visited_node[node] = True
            finished_node[node] = True
            backtrack(lines_node, temp, graph, visited_node, finished_node, node)

    # (3) filter lines with length, direction criteria
    lines = []
    for line_node in lines_node:
        num_node = len(line_node)
        if num_node == 1 : continue # 선이 아니라는 결론

        wrong = False
        line = []
        prev,cur = None,line_node[0]
        for i in range(1,num_node):
            nxt = line_node[i]
            line.extend(graph[cur][nxt])
            prev,cur = cur,nxt
        if wrong: continue
        lines.append(line)

    return lines

def backtrack(lines_node, temp, graph, visited_node, finished_node, node):
    end_pt = True
    for next_node in graph[node].keys():
        if not visited_node[next_node]:
            end_pt = False
            temp.append(next_node)
            visited_node[next_node] = True
            finished_node[next_node] = True
            backtrack(lines_node, temp, graph, visited_node, finished_node, next_node)
            del temp[-1]
            visited_node[next_node] = False
    # if there is no way to preceed, current node is the end node
    # add current line to the list
    if end_pt:
        line_node = []
        line_node.extend(temp)
        lines_node.append(line_node)

def rectify(idx):
    img_path = ''
    image = cv2.imread(img_path + 'img/image' + str(idx) +'.jpg')
    image_mask = cv2.imread(img_path + 'Mask/image' + str(idx) + '.png', cv2.IMREAD_GRAYSCALE)
    mp_hands = mp.solutions.hands

    # 7 landmark points (normalized)
    pts_index = list(range(21))
    pts_target_normalized = np.float32([[1-0.48203104734420776, 0.9063420295715332],
                                        [1-0.6043621301651001, 0.8119394183158875],
                                        [1-0.6763232946395874, 0.6790258884429932],
                                        [1-0.7340714335441589, 0.5716733932495117],
                                        [1-0.7896472215652466, 0.5098430514335632],
                                        [1-0.5655680298805237, 0.5117031931877136],
                                        [1-0.5979393720626831, 0.36575648188591003],
                                        [1-0.6135331392288208, 0.2713503837585449],
                                        [1-0.6196483373641968, 0.19251111149787903],
                                        [1-0.4928809702396393, 0.4982593059539795],
                                        [1-0.4899863600730896, 0.3213786780834198],
                                        [1-0.4894656836986542, 0.21283167600631714],
                                        [1-0.48334982991218567, 0.12900274991989136],
                                        [1-0.4258815348148346, 0.5180916786193848],
                                        [1-0.4033462107181549, 0.3581996262073517],
                                        [1-0.3938145041465759, 0.2616880536079407],
                                        [1-0.38608720898628235, 0.1775170862674713],
                                        [1-0.36368662118911743, 0.5642163157463074],
                                        [1-0.33553171157836914, 0.44737303256988525],
                                        [1-0.3209102153778076, 0.3749568462371826],
                                        [1-0.31213682889938354, 0.3026996850967407]])

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks == None: return np.zeros_like(image)
        image_height, image_width, _ = image.shape
        hand_landmarks = results.multi_hand_landmarks[0]
        pts = np.float32([[hand_landmarks.landmark[i].x*image_width,
                           hand_landmarks.landmark[i].y*image_height] for i in pts_index])
        pts_target = np.float32([[x*image_width, y*image_height] for x,y in pts_target_normalized])
        M, mask = cv2.findHomography(pts, pts_target, cv2.RANSAC,5.0)
        rectified_image = cv2.warpPerspective(image_mask, M, (image_width, image_height))
        pil_img = Image.fromarray(rectified_image)
        rectified_image = np.asarray(pil_img.resize((1024, 1024), resample=Image.NEAREST))
        return rectified_image

def get_cluster_centers(new_centers=False):
    if new_centers:
        # prepare good samples
        good = [12,104,193,212,220,249,256,295,304,396,402,487,698,908,992]
        for idx in good:
            rectified = rectify(idx)
            cv2.imwrite("good_sample/image"+str(idx)+".png",rectified)

        # put all data in feature space
        data = np.empty((0,24))
        for img_path in glob.glob("good_sample/*.png"):
            img = cv2.imread(img_path)
            skel_img = cv2.cvtColor(skeletonize(img), cv2.COLOR_BGR2GRAY)
            lines = group(skel_img)
            for line in lines:
                feature = extract_feature(line, 1024, 1024)
                data = np.vstack((data,feature))

        # k-means clustering (k=3)
        criteria = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, centers = cv2.kmeans(data.astype(np.float32), 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # sort centers according to max_y
        centers = list(centers)
        centers.sort(key = lambda x : x[2])
    else:
        centers = [np.array([5.232849  , 4.881592  , 6.3223267 , 6.64093   , 0.8113839 ,
                            0.655735  , 0.82874316, 0.74796075, 0.7993417 , 0.8345605 ,
                            0.68143266, 0.90320605, 0.5769709 , 0.9721149 , 0.53258324,
                            0.98307294, 0.4804058 , 0.9829783 , 0.36796156, 0.99141085,
                            0.24345541, 0.99082345, 0.30017138, 0.9736235 ], dtype=np.float32),
                     np.array([5.645419  , 4.169626  , 7.126243  , 6.0026045 , 0.3532842 ,
                            0.928315  , 0.4692493 , 0.9680717 , 0.578683  , 0.9680221 ,
                            0.7227269 , 0.9454175 , 0.7741767 , 0.9495983 , 0.7802345 ,
                            0.89685285, 0.8743354 , 0.8478447 , 0.85625464, 0.82669544,
                            0.88459945, 0.8000444 , 0.8956431 , 0.74734426], dtype=np.float32),
                     np.array([5.755994  , 3.8910964 , 8.680631  , 5.3926454 , 0.4247846 ,
                            0.93111324, 0.6940754 , 0.9203782 , 0.8567455 , 0.767301  ,
                            0.9177662 , 0.6054738 , 0.9801044 , 0.47111732, 0.9812451 ,
                            0.34593108, 0.97122467, 0.28715244, 0.9036454 , 0.26124895,
                            0.8069528 , 0.25324377, 0.59989274, 0.32016128], dtype=np.float32)]
    
    return centers

def save_each_line(palmline_img, lines):
    height, width, _ = palmline_img.shape
    num_lines = len(lines)
    life_img, head_img, heart_img = None, None, None

    for type_l in range(num_lines): # 3
        lines_type = lines[type_l]
        num_lines_type = len(lines_type )
        line_img = np.zeros((height, width, 3), dtype=np.uint8)
        for k in range(num_lines_type):
            line = lines_type[k]
            for y,x,_,_ in line:
                line_img[y,x] = [255, 255, 255]
        if type_l == 0:
            life_img = line_img
        elif type_l == 1:
            head_img = line_img
        elif type_l == 2:
            heart_img = line_img
    
    return life_img, head_img, heart_img

# classify lines using l2 distance with centers in feature space
def classify_lines(centers, lines, image_height, image_width, point_total):
    life_lines_dist = []
    head_lines_dist = []
    heart_lines_dist = []

    life_lines = []
    head_lines = []
    heart_lines = []

    point = point_total
    num_lines = len(lines)

    for j in range(num_lines):
        for i in range(3):
            for singleLine in lines:
                lowestdist = 1e9
                k = 0
                while k < len(singleLine):
                    temp = (point[i][0] - singleLine[k][0] ) * (point[i][0] - singleLine[k][0]) + (point[i][1] - singleLine[k][1]) * (point[i][1] - singleLine[k][1])
                    if(temp < lowestdist):
                        lowestdist = temp
                    k = k + 10

                if(i == 0):
                    life_lines_dist.append(lowestdist)
                elif(i==1):
                    head_lines_dist.append(lowestdist)
                else:
                    heart_lines_dist.append(lowestdist)


    for j in range(num_lines):
        nearest_point = 0
        if(life_lines_dist[j] > head_lines_dist[j]):
            nearest_point = 1
            if(head_lines_dist[j] > heart_lines_dist[j]):
                nearest_point = 2
        elif(life_lines_dist[j] > heart_lines_dist[j]):
                nearest_point = 2

        if(nearest_point == 0):
            life_lines.append(lines[j])
        elif(nearest_point == 1):
            head_lines.append(lines[j])
        else:
            heart_lines.append(lines[j])


    classified_lines = []
    classified_lines.append(life_lines)
    classified_lines.append(head_lines)
    classified_lines.append(heart_lines)

    return classified_lines

def classify(palmline_img, point_total):
    centers = get_cluster_centers()

    skel_img = cv2.cvtColor(skeletonize(palmline_img), cv2.COLOR_BGR2GRAY) #스켈레톤화
    cv2.imwrite('results/skel_img.jpg',skel_img)

    lines = group(skel_img)
    lines = classify_lines(centers, lines, palmline_img.shape[0], palmline_img.shape[1], point_total)
    lines = save_each_line(palmline_img, lines)

    return lines