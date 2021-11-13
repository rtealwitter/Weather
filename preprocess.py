import h5py
import numpy as np
import gc
import cv2
from visualize import plot_image

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
    #print('normalizing...')
    #normalized = ((image - np.mean(image))/np.std(image)).round(decimals=3)
    scaled = np.array(255 * (image - np.amin(image))/np.amax(image), dtype=int)
    return scaled

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
    y_scale = target_y/y_len
    x_scale = target_x/x_len
    for row in box:
        if not np.all(row==-1):
            ymin, xmin, ymax, xmax, event_class = row
            new_row = [ymin*y_scale, xmin*x_scale, ymax*y_scale, xmax*x_scale, event_class]
            box_rescaled += [new_row]
        else:
            box_rescaled += [row]
    box_rescaled = np.array(box_rescaled, dtype=int)

    return image_rescaled, box_rescaled

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
    for i in range(len(images)):
        filename = f'{foldername}/{i}.jpg'
        image_rescaled, box_rescaled = rescale(images[i], boxes[i])
        image_moved = np.moveaxis(image_rescaled, 0, -1)
        cv2.imwrite(filename, image_moved)
        boxes_rescaled += [box_rescaled]     

    with open(f'{foldername}/bboxes.txt', 'w') as f:
        for box in boxes_rescaled:
            f.write(str(box.tolist())+'\n')

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
    #index = 4
    #img, bx = rescale(images[index], boxes[index])
    #plot_image(img, bx)

    year = '1979'
    hours = list(range(1460)[4*181:4*(181+31)])
    weather_variables = [2,6,10]
    shrink_data(year, weather_variables, foldername='train', hours=hours)
    #images, boxes = read_data(year)
    #print(images.shape)
    #plotbox(image, box)


