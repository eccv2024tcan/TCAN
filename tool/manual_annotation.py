import cv2
import numpy as np
import math
from PIL import Image
import pickle
import os
import argparse

# Define the number of keypoints
NUM_KEYPOINTS = 18

# Define colors
GREEN = (0, 255, 0)

# Initialize the dictionary for keypoints
dwpose = {
    'bodies': {
        'candidate': np.zeros((NUM_KEYPOINTS, 2)),
        'subset': np.zeros((1, NUM_KEYPOINTS))
    },
    'hands': {}
}


# copy from https://github.dev/IDEA-Research/DWPose > ControlNet_v1_1_nightly/annotator/dwpose/util.py
def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas

def save_pose_as_img(save_path, dwpose, H, W):
    candidate = dwpose['bodies']['candidate']
    subset = dwpose['bodies']['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = draw_bodypose(canvas, candidate, subset)
    
    Image.fromarray(canvas).save(save_path) 

def save_as_json(save_path, dwpose):
    with open(save_path, 'wb') as fp:        
        pickle.dump(dwpose, fp)

# Function to handle mouse click events
def mouse_click(event, x, y, flags, param):
    global index 
    if event == cv2.EVENT_LBUTTONDOWN:
        # Normalize the coordinates
        normalized_x = x / image_width
        normalized_y = y / image_height

        # Update the candidate keypoints
        dwpose['bodies']['candidate'][index] = [normalized_x, normalized_y]

        # Draw a circle at the clicked point
        cv2.circle(image, (x, y), 3, GREEN, -1)
        cv2.imshow('Image', np.concatenate([image, image_ref], axis=1))

        # Increment index
        # param['index'] += 1
        index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Define PATH
    parser.add_argument("--PATH_REF", default='inputs/webtoon_characters/openpose.png')
    parser.add_argument("--DIRPATH_INPUT_IMAGE", default='inputs/webtoon_characters/source_images')
    parser.add_argument("--DIRPATH_OUTPUT_POSE", default='inputs/webtoon_characters/openpose')
    parser.add_argument("--DIRPATH_OUTPUT_JSON", default='inputs/webtoon_characters/openpose_dict')

    args = parser.parse_args()
    
    os.makedirs(args.DIRPATH_OUTPUT_POSE, exist_ok=True)
    os.makedirs(args.DIRPATH_OUTPUT_JSON, exist_ok=True)

    file_list = os.listdir(args.DIRPATH_INPUT_IMAGE)
    
    for file_name in file_list:
        path_input_image = os.path.join(args.DIRPATH_INPUT_IMAGE, file_name)    

        # Read the image
        image = cv2.imread(path_input_image)

        # Get the dimensions of the image
        image_height, image_width, _ = image.shape

        # Get Reference Image
        image_ref = cv2.imread(args.PATH_REF)
        image_ref = cv2.resize(image_ref, (image_width, image_height))

        # Display the image
        cv2.imshow('Image', np.concatenate([image, image_ref], axis=1))

        # Initialize index
        index = 0

        # Set mouse callback function        
        cv2.setMouseCallback('Image', mouse_click)

        # Wait for user to annotate keypoints
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if index == NUM_KEYPOINTS:
                break

        # Set the subset indices
        dwpose['bodies']['subset'][0] = np.arange(NUM_KEYPOINTS)

        # Print the result
        print(dwpose)
        save_pose_as_img(os.path.join(args.DIRPATH_OUTPUT_POSE, file_name), dwpose, image_height, image_width)
        save_as_json(os.path.join(args.DIRPATH_OUTPUT_JSON, file_name.split('.')[0]+'.pkl'), dwpose)

        # Destroy all OpenCV windows
        cv2.destroyAllWindows()