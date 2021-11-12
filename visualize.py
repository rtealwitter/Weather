from matplotlib import patches
import matplotlib.pyplot as plt

EVENTS = ['Tropical Depression', 'Tropical Cyclone', 'Extratropical Cyclone', 'Atmospheric River']

NAMES = ['Precipitation', 'Surface Pressure', 'Sea Level Pressure', 
         'Reference Humidity', 'Temperature at 200 mbar', 'Temperature at 500 mbar',
         'Total Water', 'Reference Temperature', 'Surface Temperature',
         'Zonal Wind', 'Lowest Zonal Wind', 'Meridional Wind',
         'Lowest Meridional Wind', 'Geopotential at 100 mbar',
         'Geopotential at 200 mbar', 'Lowest Model Height']

def plot_image(image, box):
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
