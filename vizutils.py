import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

# def implt(img, color=None, title=''):
#     """Show image using plt."""
#     plt.imshow(img, cmap=color)
#     plt.title(title)
#     plt.show()
    
#     return


def is_color(img):
    if img.ndim <=2:
        return False

    # img.ndim >=3
    if img.shape[-1] == 1:
        return False

    return True


def show_img(img, color='gray', is_bgr=False, title='', figsize=None):
    """Show image using plt."""
    

    if figsize:
        if not isinstance(figsize, (tuple, list)):
            figsize = (figsize, figsize)
        plt.figure(figsize=figsize)

    if is_color(img) and is_bgr:
        img = img[...,::-1]

    params = {'cmap':color}

    if img.dtype == np.uint8:
        params.update({'vmin': 0, 'vmax': 255})

    plt.imshow(img, **params)
    plt.title(title)
    plt.show()
    
    return
    

def imgrid(imgs, rows, cols, figsize=(30,30), save_name = None):

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(rows, cols, figure=fig, wspace=0, hspace=0)
#     grid.update() 

    for i,cell in enumerate(grid):
#         print(i, cell)
        
        plt.subplot(cell)
        plt.imshow(imgs[i], 'gray')

    if not save_name is None:
        plt.savefig(f'{save_name}.png', bbox_inches = 'tight', pad_inches = 0)
    
    return

