import csv
import numpy as np


def load_csv():
    with open('train_image.csv', newline='') as train_image_file:
        train_images = np.array(list(csv.reader(train_image_file)), dtype='float64')
    with open('train_label.csv', newline='') as train_label_file:
        train_labels = np.array(list(csv.reader(train_label_file)), dtype='int32')
    with open('test_image.csv', newline='') as test_image_file:
        test_images = np.array(list(csv.reader(test_image_file)), dtype='float64')
    # with open('test_label.csv', newline='') as test_label_file:
    #     test_labels = np.array(list(csv.reader(test_label_file)), dtype='int32')
    return train_images, train_labels, test_images
