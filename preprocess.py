import h5py
import numpy as np

NAMES = ['Precipitation', 'Surface Pressure', 'Sea Level Pressure', 
         'Reference Humidity', 'Temperature at 200 mbar', 'Temperature at 500 mbar',
         'Total Water', 'Reference Temperature', 'Surface Temperature',
         'Zonal Wind', 'Lowest Zonal Wind', 'Meridional Wind',
         'Lowest Meridional Wind', 'Geopotential at 100 mbar',
         'Geopotential at 200 mbar', 'Lowest Model Height']

EVENTS = ['Tropical Depression', 'Tropical Cyclone', 'Extratropical Cyclone', 'Atmospheric River']

# Weather variables:
# Total (convective and large-scale) precipitation rate (liq + ice)
# Surface Pressure
# 2. Sea level pressure
# Reference height humidity
# 4. Temperature at 200 mbar pressure surface
# 5. Temperature at 500 mbar pressure surface
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

from matplotlib import patches
import matplotlib.pyplot as plt

def plot_image(image, box):
    ## warning: bug for one channel rescaled box
    plt.rcParams.update({'font.size': 7})
    if len(image.shape) == 2: image = np.array([image])
    num_channels = image.shape[0]
    num_rows = int(np.sqrt(num_channels))
    num_cols = int(np.round(num_channels//num_rows))
    fig, axs = plt.subplots(num_rows,num_cols) 
    fig.tight_layout()
    for i in range(num_rows):
        for j in range(num_cols):
            channel = i*num_cols + j 
            if channel < image.shape[0]:
                if num_channels == 1: ax = axs
                elif num_rows == 1: ax = axs[j]
                elif num_rows > 1: ax = axs[i,j]
                ax.axis('off')
                ax.imshow(image[channel,])                
                addbox(ax, box)
                ax.set_title(NAMES[channel])
    plt.show()
    
def addbox(ax, box):
    for row in box:
        if not np.all(row == -1):
            ymin, xmin, ymax, xmax, event_class = row
            ax.add_patch(patches.Rectangle(xy=(xmin, ymin),width=xmax-xmin, height=ymax-ymin, fill=False, label=EVENTS[event_class]))

def shrink_data(year, hours, weather_variables):
    large_path = f'h5data/climo_{year}.h5'
    small_path = f'h5small/climo_{year}.h5'

    with h5py.File(large_path, 'r') as large_file:
        images = large_file["images"][hours,]
        images = images[:,weather_variables,]
        boxes = large_file["boxes"][hours,]

    with h5py.File(small_path, 'w') as small_file:
        small_file.create_dataset(name="images",data=images, shape=images.shape, dtype='f4', compression="gzip")
        small_file.create_dataset(name="boxes",data=boxes, shape=boxes.shape, dtype='i4', compression="gzip")

def read_data(year):
    data_path = f'h5small/climo_{year}.h5'
    with h5py.File(data_path, 'r') as data_file:       
        return data_file["images"][:,], data_file["boxes"][:,]


year = "1979"
#random_indices = sorted(np.random.randint(0,1460, size=(1,10)).tolist()[0])
#sample_data(year, random_indices)
#plot_all_channels(year, 4)

import cv2

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
    print(image.shape)
    print(image_rescaled.shape)

    # Rescale boxes
    box_rescaled = []
    y_scale = target_y/y_len
    x_scale = target_x/x_len
    for row in box:
        if not np.all(row==-1):
            ymin, xmin, ymax, xmax, event_class = row
            new_row = [ymin*y_scale, xmin*x_scale, ymax*y_scale, xmax*x_scale, event_class]
            box_rescaled += [new_row]
    box_rescaled = np.array(box_rescaled, dtype=int)

    plot_image(image_rescaled, box_rescaled)
    #plot_image(image, box) 
    return image_rescaled, box_rescaled


images, boxes = read_sample('1979')
index = 2
rescale(images[index], boxes[index])
#plot_one_sample(year)

#hours = list(range(1460)[4*181:4*(181+31)])
#weather_variables = [2,4,5,6,10]
#shrink_data(year, hours, weather_variables)
#images, boxes = read_data(year)
#print(images.shape)
#plotbox(image, box)


