import cv2
import os.path as path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
import warnings
import sys
import math
from os import listdir
from os.path import isfile, join


def log_img(img_name, img_data):
    # pass
    cv2.imwrite(log_path(img_name), img_data)

# 图片日志路径
def log_path(img_name):
    return path.join(path.dirname(__file__),  'log', '{}.jpg'.format(img_name))


# 图片输入路径
def input_path(img_name):
    return path.join(path.dirname(__file__), 'source', '{}.jpg'.format(img_name))


def color_path(img_name):
    return path.join(path.dirname(__file__), 'color', '{}.jpg'.format(img_name))


# 图片输出路径
def output_path(img_name):
    return path.join(path.dirname(__file__), 'recognize', 'source', 'output', '{}.jpg'.format(img_name))


#  LAB 色差计算
def color_euclidean_distance(color_a, color_b, **kwargs):
    a = bgr_to_lab_color(color_a)
    b = bgr_to_lab_color(color_b)
    percent = kwargs.get('percent', 0)
    return np.linalg.norm(a - b)


#  https://www.compuphase.com/cmetric.htm
#  low-cost approximation
# def color_euclidean_distance(color_a, color_b, **kwargs):
#     ab, ag, ar = np.array(color_a, dtype=int)
#     bb, bg, br = np.array(color_b, dtype=int)
#     rmean = (br + ar)*0.5
#     r = ar - br
#     g = ag - bg
#     b = ab - bb
#     return math.sqrt((((512 + rmean) * (r**2))/256) + 4*(g**2) + (((767 - rmean) * (b**2))/256))


def scale_img(img, size):
    return cv2.resize(img, size)


def kron_img(des, crouton_imgs, crouton_size):
    (des_width, des_height, _) = des.shape
    des_matrix = lambda n: np.full((des_width, des_height), n)
    crouton_imgs = list(map(lambda d: scale_img(d, crouton_size), crouton_imgs))
    (width, height) = crouton_size
    color_pos_toast = np.kron(des_matrix(1), np.arange(0, width * height).reshape(width, height))
    color_percent_list = color_percent_from_photos(crouton_imgs)
    # for index, c in enumerate(color_percent_list):
    #     log_img('test{}.png'.format(index), np.full((100, 100, 3), c['color']))
    crouton_index_snap = photo_snap_of_color_snap(des, color_percent_list)
    crouton_index_toast = np.kron(crouton_index_snap, np.full(crouton_size, 1))
    (res_width, res_height) = crouton_index_toast.shape
    buffer_img = np.empty((res_width, res_height, 3), dtype=np.uint8)

    for r in range(0, res_width):
        for c in range(0, res_height):
            crouton_img = crouton_imgs[crouton_index_toast[r][c]]
            color_pos = color_pos_toast[r][c]
            buffer_img[r][c] = crouton_img[color_pos // height][color_pos % height]
    return buffer_img


def mosaic_img(img, **kwargs) -> (np.ndarray, np.ndarray):
    blur_size = kwargs.get('blur_size', None)
    blur_width = kwargs.get('blur_width', None)
    blur_height = kwargs.get('blur_height', None)
    if blur_size is not None:
        blur_width = blur_size
        blur_height = blur_size
    if blur_width is None or blur_height is None:
        raise BaseException('blur_width or blur_height is None')
    buffer_img = img.copy()
    (width, height, _) = buffer_img.shape

    snapsize = width // blur_width + 1, \
               height // blur_height + 1, \
               _
    snapshoot = np.empty(snapsize, dtype=np.uint8)
    for row in range(0, width):
        for col in range(0, height):
            r = row - (row % blur_width) + blur_width // 2
            c = col - (col % blur_height) + blur_height // 2
            if r >= width:
                r = (row - (row % blur_width) + (width - 1)) // 2
            if c >= height:
                c = (col - (col % blur_height) + (height - 1)) // 2
            buffer_img[row][col] = buffer_img[r][c]
            snapshoot[row // blur_width][col // blur_height] = buffer_img[r][c]

    return buffer_img, snapshoot


def photo_snap_of_color_snap(color_snap, color_percent_list):
    (width, height, _) = color_snap.shape
    crouton_index_snap = np.full((width, height), 0)
    for r in range(0, width):
        for c in range(0, height):
            color_a = color_snap[r][c]
            min_dist = sys.maxsize
            min_index = -1
            for color_percent in color_percent_list:
                color_b = color_percent['color']
                percent = color_percent['percent']
                index = color_percent['index']
                distance = color_euclidean_distance(color_a, color_b, percent=percent)
                if distance < min_dist:
                    min_dist = distance
                    min_index = index
            else:
                if min_index != -1:
                    crouton_index_snap[r][c] = min_index
    return crouton_index_snap


def color_percent_from_photos(imgs):
    color_percent_list = []
    for (index, img) in enumerate(imgs):
        r = color_percent_from_photo(img)
        color_percent_list += list(map(lambda d: {**d, 'index': index}, r))
    return color_percent_list


def color_percent_from_photo(img) -> list:
    buffer_img, _ = mosaic_img(img, blur_size=15)
    (width, height, alpha) = buffer_img.shape
    color_point = 3
    print('开始聚类...')
    X = np.array(buffer_img.reshape((width * height, alpha)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        kmeans = KMeans(n_clusters=color_point, random_state=0).fit(X)
    print('聚类完成')
    center = kmeans.cluster_centers_.astype(np.uint8)
    labels_ = kmeans.labels_
    for poi in range(0, width * height):
        row = poi // height
        col = poi % height
        buffer_img[row][col] = center[labels_[poi]]
    unique, counts = np.unique(labels_, return_counts=True)
    r = []
    for k, v in dict(zip(unique, counts)).items():
        r.append({
            'color': center[k],
            'percent': v / (width * height)
        })
    # return r
    return [*sorted(r, key=lambda d: d['percent'])]

def imread(image_path):
    return cv2.imread(image_path)


def bgr_to_lab_color(bgr_color):
    b = np.array([[
        bgr_color
    ]], dtype=np.uint8)
    lab_color = cv2.cvtColor(b, cv2.COLOR_BGR2Lab).astype(np.float64)[0][0]
    lab_color[0] *= 100 / 255
    lab_color[1] -= 128
    lab_color[2] -= 128
    return lab_color

def main():
    image_path = input_path('aim')
    image = imread(image_path)
    mosaic, snapshoot = mosaic_img(image, blur_size=2)

    log_img('mosaic', mosaic)
    log_img('snap', snapshoot)
    color_dir = path.join(path.dirname(__file__), 'color')
    img_names = [path.splitext(f)[0] for f in listdir(color_dir) if isfile(join(color_dir, f))]
    end_img = kron_img(snapshoot, [imread(color_path(name)) for name in img_names], (20, 20))
    log_img('end', end_img)


main()

# def test():
#     c_list = [
#         [223, 198, 255],
#         [230,  88,  62],
#         [24,   22, 18],
#         [ 19, 167, 245],
#         [ 31, 234, 108],
#         [254, 254, 254],
#         [  7,   1, 142],
#
#
#         [255, 204, 153],
#     ]
#
#     aim_c = [241, 196, 135]
#
#     for c in c_list:
#         a = bgr_to_lab_color(c)
#         b = bgr_to_lab_color(aim_c)
#         # print(a)
#         # print(b)
#         print(color_euclidean_distance(c, aim_c))
#         print('**********************')


# test()

# image_path = color_path('pink')
# image1 = cv2.imread(image_path)

#  223 198 255
# 218 152 123

# a = np.array([
#     [1, 2],
#     [1, 1]
# ])
#
# b = np.full((20, 20), 1)
# b_ = np.kron(np.array([
#     [1, 1],
#     [1, 1]
# ]), np.arange(0, 20*20).reshape((20, 20)))
# c = np.kron(a, b)
#
# print(c)
