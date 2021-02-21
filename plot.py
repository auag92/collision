import matplotlib.pyplot as plt


def draw_1d(x, y, title=""):
    plt.plot(x, y)
    plt.title(title)
    plt.grid(b=None)
    plt.show()


def draw_1d_by_side(x, y1, y2):
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(15,5))
    axs[0].plot(x, y2)
    axs[1].plot(x, y1)
    for ax in axs:
        ax.grid(False)
    plt.show()


def draw_2d(im, title=None):
    """
    plot a 2d image
    """
    plt.imshow(im, cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.show()
    
    
def draw_2d_by_side(im1, im2, title=None):
    """
    plot a 2d image
    """
    fig, axs = plt.subplots(1, 2, figsize=[12, 6])
    axs[0].imshow(im1, cmap="viridis")
    axs[1].imshow(im2, cmap="viridis")
    for idx, ax in enumerate(fig.get_axes()):
        ax.label_outer()
#         if idx > 0:
        ax.axis('off')
#     plt.grid('False')
    plt.title(title)
    plt.show()