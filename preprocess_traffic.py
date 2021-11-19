import numpy as np
import cv2
import os
from visualize import plot_image_traffic

def rescale(image, bboxes, target_x=300, target_y=300):
    # Rescale all channels
    y_len, x_len, _ = image.shape
    image_rescaled = cv2.resize(image, (target_x, target_y))

    # Rescale boxes
    box_rescaled = []
    y_scale = target_y/y_len
    x_scale = target_x/x_len
    for box in bboxes:
        xmin, ymin, xmax, ymax = box
        new_row = [int(xmin*x_scale), int(ymin*y_scale), int(xmax*x_scale), int(ymax*y_scale)]
        box_rescaled += [new_row]

    return image_rescaled, box_rescaled

def extract_annotations(filename, image):
    labels = []
    bboxes = []
    xscale, yscale = image.shape[1], image.shape[0]
    with open(filename, 'r') as f:
        for line in f:
            label, xcenter, ycenter, width, height = [float(i) for i in line.split()]
            xcenter, ycenter, width, height = xcenter*xscale, ycenter*yscale, width*xscale, height*yscale
            xmin, ymin, xmax, ymax = xcenter - width/2, ycenter - height/2, xcenter + width/2, ycenter + height/2
            labels += [int(label)-1]
            bboxes += [[int(val) for val in [xmin, ymin, xmax, ymax]]]

    image, bboxes = rescale(image, bboxes)
    return image, bboxes, labels

def preprocess():
    np.random.seed(1)
    train_num, valid_num, test_num = 0, 0, 0
    unique_labels = set()

    for use in ['train', 'valid', 'test']:
        folder_save = 'traffic/' + use + '/'
        for f in os.listdir(folder_save):
            os.remove(folder_save+f)

    for condition in ['Fog', 'Rain', 'Sand', 'Snow']:
        folder = 'DAWN/'+condition
        folder_box = folder + '/' + condition + '_YOLO_darknet/'

        files = [f for f in os.listdir(folder) if os.path.isfile(folder+'/'+f)]
        for image_file in files:
            image = cv2.imread(folder+'/'+image_file)
            box_file = image_file[:-3] + 'txt'
            image, bboxes, labels = extract_annotations(folder_box+box_file, image)
            unique_labels = unique_labels.union(set(labels))


            use = np.random.choice(['train', 'valid', 'test'], p=[.8, .1, .1])

            folder_save = 'traffic/' + use + '/'
            if len(bboxes) > 0:
                if use == 'train':
                    filename = folder_save + str(train_num)+'.jpg'
                    train_num += 1
                if use == 'valid':
                    filename = folder_save + str(valid_num)+'.jpg'
                    valid_num += 1
                if use == 'test':
                    filename = folder_save + str(test_num)+'.jpg'
                    test_num += 1

                cv2.imwrite(filename, image)
                with open(folder_save + 'labels.txt', 'a') as f:
                    f.write(str(labels) + '\n')
                with open(folder_save + 'bboxes.txt', 'a') as f:
                    f.write(str(bboxes) + '\n')
    
    print(unique_labels)


if __name__ == '__main__':
    preprocess()
