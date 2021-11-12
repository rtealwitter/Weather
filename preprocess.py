import h5py
import numpy as np
import gc
import cv2

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

def rescale(image, box, target_x=300, target_y=300):
    # Rescale all channels
    if len(image.shape) == 3:
        num_channels, y_len, x_len = image.shape
        image_rescaled = []
        for i in range(num_channels):
            image_rescaled += [cv2.resize(image[i,], (target_x, target_y))]
        image_rescaled = np.array(image_rescaled)
    elif len(image.shape) == 2:
        y_len, x_len = image.shape
        image_rescaled = cv2.resize(image, (target_x, target_y))

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

def shrink_data(year, hours, weather_variables):
    large_path = f'h5data/climo_{year}.h5'
    small_path = f'h5small/climo_{year}.h5'

    # Load data in hours and weather_variables
    with h5py.File(large_path, 'r') as large_file:
        images = large_file["images"][hours,]
        images = images[:,weather_variables,]
        boxes = large_file["boxes"][hours,]
    
    # Rescale to 300x300
    images_rescaled = []
    boxes_rescaled = []
    for i in range(len(images)):
        image_rescaled, box_rescaled = rescale(images[i], boxes[i])
        images_rescaled += [image_rescaled]
        boxes_rescaled += [box_rescaled] 
    images_rescaled = np.array(images_rescaled, dtype=int)
    boxes_rescaled = np.array(boxes_rescaled, dtype=int)

    with h5py.File(small_path, 'w') as small_file:
        small_file.create_dataset(name="images",data=images_rescaled, shape=images_rescaled.shape, dtype='f4', compression="gzip")
        small_file.create_dataset(name="boxes",data=boxes_rescaled, shape=boxes_rescaled.shape, dtype='i4', compression="gzip")

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
    #plot_one_sample(year)

    #year = '1979'
    #hours = list(range(1460)[4*181:4*(181+31)])
    #weather_variables = [2,6,10]
    #shrink_data(year, hours, weather_variables)
    #images, boxes = read_data(year)
    #print(images.shape)
    #plotbox(image, box)


