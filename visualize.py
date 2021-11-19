from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np

LABELS = ['Tropical Depression', 'Tropical Cyclone', 'Extratropical Cyclone', 'Atmospheric River']

NAMES = ['Precipitation', 'Surface Pressure', 'Sea Level Pressure', 
         'Reference Humidity', 'Temperature at 200 mbar', 'Temperature at 500 mbar',
         'Total Water', 'Reference Temperature', 'Surface Temperature',
         'Zonal Wind', 'Lowest Zonal Wind', 'Meridional Wind',
         'Lowest Meridional Wind', 'Geopotential at 100 mbar',
         'Geopotential at 200 mbar', 'Lowest Model Height']

def plot_image_extreme(image, box=None, labels=None):
  if len(image.shape) == 2: image = np.array([image])
  # Adapt subplot to the number of channels
  num_channels = image.shape[0]
  num_rows = int(np.sqrt(num_channels))
  num_cols = int(np.round(num_channels//num_rows))
  fig, axs = plt.subplots(num_rows,num_cols, figsize=(15,15)) 
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
        if box != None:
          addbox(ax, box, labels)
        ax.set_title(NAMES[channel])
  if box != None:
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.show()
    
def addbox(ax, box, labels):
  for i in range(len(box)):
    left, bottom, right, top = box[i]
    handles, existing_labels = ax.get_legend_handles_labels()
    label = "" if (labels == None) or (LABELS[int(labels[i])] in existing_labels) else LABELS[int(labels[i])]
    ax.add_patch(patches.Rectangle(xy=(left, bottom), width=right-left,
                                    height=top-bottom, fill=False,
                                    label=label))

def plot_image_traffic(image, bboxes, labels):
    _, ax = plt.subplots()
    ax.imshow(image)
    for i in range(len(bboxes)):
        left, bottom, right, top = bboxes[i]
        _, existing_labels = ax.get_legend_handles_labels()
        label = "" if (labels == None) or (str(labels[i]) in existing_labels) else labels[i]
        ax.add_patch(patches.Rectangle(xy=(left, bottom), width=right-left,
                                        height=top-bottom, fill=False,
                                        label=label))
    plt.legend()                                    
    plt.show()

