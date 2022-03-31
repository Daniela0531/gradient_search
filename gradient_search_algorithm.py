import typing
import numpy as np
import cv2 as cv

topmost_bound = 75
Point = typing.Tuple[int, int]
Rectangle = typing.Tuple[Point, Point]

def gradient_search_algorithm(img: np.ndarray):
    #consider the gradient through derivatives by definition
    derivative_x = (img[1:, :] - img[:-1, :])[:, :-1]
    derivative_y = (img[:, 1:] - img[:, :-1])[:-1:, :]
    gradient_img = np.sqrt(
        np.linalg.norm(derivative_x, axis=-1) ** 2
        + np.linalg.norm(derivative_y, axis=-1) ** 2,
        )
    the_largest_rectangle = None

    neighboring_up = -(np.ones(gradient_img.shape[1]).astype(int))
    neighboring_left = -(np.ones(gradient_img.shape[1]).astype(int))
    neighboring_right = (np.ones(gradient_img.shape[1]).astype(int) * gradient_img.shape[1])

    cv.imshow("window_name", gradient_img)
    #looking for a submatrix with values not higher than the threshold
    max_square = 0
    stack = []
    for i in range(gradient_img.shape[0]):
        #dynamics precalculation
        for j in range(gradient_img.shape[1]):
            if gradient_img[i][j] > topmost_bound:
                neighboring_up[j] = i
        stack.clear()
        #choose the left border  submatrix
        for j in range(gradient_img.shape[1]):
            while len(stack) != 0 and neighboring_up[stack[-1]] <= neighboring_up[j]:
                stack.pop()
            if len(stack) != 0:
                neighboring_left[j] = stack[-1]
            stack.append(j)
        stack.clear()
        #choose the right border submatrix
        for j in reversed(range(gradient_img.shape[1])):
            while len(stack) != 0 and neighboring_up[stack[len(stack) - 1]] <= neighboring_up[j]:
                stack.pop()
            if len(stack) != 0:
                neighboring_right[j] = stack[-1]
            stack.append(j)
        #calculate the maximum square
        for j in range(gradient_img.shape[1]):
            square = (i - neighboring_up[j] - 1)*(neighboring_right[j] - 1 - neighboring_left[j] - 1)
            if square > 0 and square > max_square :
                max_square = square
                the_largest_rectangle = (
                    (neighboring_up[j] + 1, neighboring_left[j] + 1),
                    (i, neighboring_right[j] - 1),
                )
    return the_largest_rectangle
