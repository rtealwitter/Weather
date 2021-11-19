import h5py
import numpy as np
import gc
import cv2
from visualize import plot_image_extreme

# Weather variables:
# Total (convective and large-scale) precipitation rate (liq + ice)
# Surface Pressure
# 2. Sea level pressure
# Reference height humidity
# Temperature at 200 mbar pressure surface
# Temperature at 500 mbar pressure surface
# 6. Total (vertically integrated) precipitatable water
# Reference height temperature
# Surface temperature (radiative)
# Zonal wind at 850 mbar pressure surface
# 10. Lowest model level zonal wind
# Meridional wind at 850 mbar pressure surface
# Lowest model level meridional wind
# Geopotential Z at 100 mbar pressure surface
# Geopotential Z at 200 mbar pressure surface
# Lowest model level height

def normalize(image):
    return np.array(255 * (image - np.amin(image))/np.amax(image), dtype=int)

def rescale(image, box, target_x=300, target_y=300):
    # Rescale all channels
    if len(image.shape) == 3:
        num_channels, y_len, x_len = image.shape
        image_rescaled = []
        for i in range(num_channels):
            image_rescaled += [normalize(cv2.resize(image[i,], (target_x, target_y)))]
        image_rescaled = np.array(image_rescaled)
    elif len(image.shape) == 2:
        y_len, x_len = image.shape
        image_rescaled = normalize(cv2.resize(image, (target_x, target_y)))

    # Rescale boxes
    box_rescaled = []
    labels = []
    y_scale = target_y/y_len
    x_scale = target_x/x_len
    for row in box:
        if not np.all(row==-1):
            ymin, xmin, ymax, xmax, event_class = row
            labels += [event_class]
            # ltrb (left top right bottom) format
            new_row = [int(xmin*x_scale), int(ymin*y_scale), int(xmax*x_scale), int(ymax*y_scale)]
            box_rescaled += [new_row]

    return image_rescaled, box_rescaled, labels

def sample_data(year, indices):
    data_path = f'h5data/climo_{year}.h5'
    sample_path = f'h5small/sample_{year}.h5'
    with h5py.File(data_path, 'r') as data_file:
        images = data_file['images'][indices,]
        boxes = data_file['boxes'][indices,]
    
    with h5py.File(sample_path, 'w') as sample_file:
        sample_file.create_dataset(name='images', data=images, compression='gzip')
        sample_file.create_dataset(name='boxes', data=boxes, compression='gzip')

def read_sample(year):
    sample_path = f'h5small/sample_{year}.h5'
    with h5py.File(sample_path, 'r') as sample_file:
        return sample_file['images'][:,], sample_file['boxes'][:,]

def shrink_data(year, weather_variables, foldername, hours=None):
    large_path = f'h5data/climo_{year}.h5'
    #small_path = f'h5small/climo_{year}.h5'

    # Load data in hours and weather_variables
    with h5py.File(large_path, 'r') as large_file:
        images = large_file["images"][:,weather_variables,]
        if hours != None:
            images = images[hours,]
            boxes = large_file["boxes"][hours,]
        else:
            boxes = large_file["boxes"][:]
    
    # Rescale to 300x300
    boxes_rescaled = []
    labels = []
    current_num = 0
    for i in range(len(images)):
        filename = f'{foldername}/{current_num}.jpg'
        image_rescaled, box_rescaled, label = rescale(images[i], boxes[i])
        if len(box_rescaled) > 0:
            image_moved = np.moveaxis(image_rescaled, 0, -1)
            cv2.imwrite(filename, image_moved)
            boxes_rescaled += [box_rescaled]     
            labels += [label]
            current_num += 1

    with open(f'{foldername}/labels.txt', 'w') as f:
        for label in labels:
            f.write(str(label)+'\n')

    with open(f'{foldername}/bboxes.txt', 'w') as f:
        for box in boxes_rescaled:
            f.write(str(box)+'\n')

def read_data(year):
    data_path = f'h5small/climo_{year}.h5'
    with h5py.File(data_path, 'r') as data_file:       
        return data_file["images"][:,], data_file["boxes"][:,]

if __name__ == '__main__':
    gc.collect()

    #year = "1979"
    #random_indices = sorted(np.random.randint(0,1460, size=(1,10)).tolist()[0])
    #sample_data(year, random_indices)

    #images, boxes = read_sample('1979')
    #index = 5
    #img, bx, labels = rescale(images[index], boxes[index])
    #print(len(bx))
    #plot_image_extreme(img, bx, labels)

    year = '1979'
    weather_variables = [2,6,10]
    nov1 = 4*(365-61)
    dec1 = 4*(365-31)
    hours_train = list(range(1460)[:nov1])
    shrink_data(year, weather_variables, foldername='extreme/train', hours=hours_train)

    hours_valid = list(range(1460)[nov1:dec1])
    shrink_data(year, weather_variables, foldername='extreme/valid', hours=hours_valid)

    hours_test = list(range(1460)[dec1:])
    shrink_data(year, weather_variables, foldername='extreme/test', hours=hours_test)



