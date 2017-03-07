import cv2
import numpy as np
import pickle
import random
import shared

rows = 10
columns = 25
total = rows * columns

with open('../data/train.p', mode='rb') as f:
    train = pickle.load(f)

features = train['features']

sample = np.array(list(map(lambda x: features[x], random.sample(range(len(features)), total))))
collage = cv2.cvtColor(shared.collage(sample, rows, columns), cv2.COLOR_RGB2BGR)

processed_sample = np.array(list(map(lambda x: shared.gray_normalized(x), sample)))
processed_collage = shared.collage(processed_sample, rows, columns)

cv2.imwrite('../images/dataset_sample.png', collage)
cv2.imwrite('../images/dataset_sample_processed.png', processed_collage)

cv2.imshow('collage', collage)
cv2.waitKey(0)

cv2.imshow('processed_collage', processed_collage)
cv2.waitKey(0)

cv2.destroyAllWindows()
