import cv2
import math
import numpy as np
import pickle
import random
import shared

# Based on http://stackoverflow.com/a/19110462, need a better way to calc transformation matrix...
def warp(img, rotx=0, roty=0, rotz=0, f=2):
    h, w, c = img.shape

    cx = math.cos(rotx)
    sx = math.sin(rotx)
    cy = math.cos(roty)
    sy = math.sin(roty)
    cz = math.cos(rotz)
    sz = math.sin(rotz)

    roto = [
        [cz * cy, cz * sy * sx - sz * cx],
        [sz * cy, sz * sy * sx + cz * cx],
        [-sy, cy * sx]
    ]

    pt = [
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ]

    ptt = np.zeros((4, 2), dtype=float)

    for i in range(4):
        pz = pt[i][0] * roto[2][0] + pt[i][1] * roto[2][1]
        ptt[i][0] = w / 2 + (pt[i][0] * roto[0][0] + pt[i][1] * roto[0][1]) * f * h / (f * h + pz)
        ptt[i][1] = h / 2 + (pt[i][0] * roto[1][0] + pt[i][1] * roto[1][1]) * f * h / (f * h + pz)

    src = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    dst = np.float32([
        [ptt[0][0], ptt[0][1]],
        [ptt[1][0], ptt[1][1]],
        [ptt[2][0], ptt[2][1]],
        [ptt[3][0], ptt[3][1]]
    ])

    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)

with open('../data/train.p', mode='rb') as f:
    train = pickle.load(f)

features = train['features']

angle = 30
rows = 10
columns = 25

sample = np.array(list(map(lambda x: features[x], random.sample(range(len(features)), rows))))
combinations = np.random.randint(-angle, angle, (columns - 1, 3)) * math.pi / 180

augmented = []

for sign in sample:
    augmented.append(sign)
    for rx, ry, rz in combinations:
        augmented.append(warp(sign, rx, ry, rz))

collage = cv2.cvtColor(shared.collage(np.array(augmented), rows, columns), cv2.COLOR_RGB2BGR)

cv2.imwrite('../images/dataset_sample_augumented.png', collage)

cv2.imshow('collage', collage)
cv2.waitKey(0)

cv2.destroyAllWindows()