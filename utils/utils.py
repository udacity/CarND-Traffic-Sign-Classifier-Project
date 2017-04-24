# coding=utf-8
from csv import DictReader

def get_classes(f_name):
    classes = []
    with open(f_name, 'rt') as f:
        reader = DictReader(f, delimiter=',')
        for row in reader:
            classes.append((row['SignName']))
    return classes
