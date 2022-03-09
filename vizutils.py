import matplotlib.pyplot as plt
from matplotlib import gridspec


def implt(img, color=None, title=''):
    """Show image using plt."""
    plt.imshow(img, cmap=color)
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

