import os.path
import sys
import numpy as np
from gradient_search_algorithm import gradient_search_algorithm
import cv2 as cv


def main():
    if len(sys.argv) != 2 :
        print("folder path not specified")
        return -1
    src_path = sys.argv[1]
    if not os.path.exists(src_path) :
        print("you specified a non-existent folder")
        return -1
    img = cv.imread(src_path)
    img = img.astype(np.float64)

    max_rectangle = gradient_search_algorithm(img)
    print(max_rectangle)

if __name__ == '__main__':
    main()