import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection


def wedge_plot(ax, data, index, data_label, num_regions, num_zones, color_palette, start_degree, end_degree, show_index=True):
    """
    Generate a wedge plot with data, index, and labels.

    Parameters:
        ax (matplotlib.axes.Axes): The axes object to plot the wedges on.
        data (numpy.ndarray): 2D array of data values for coloring the wedges.
        index (numpy.ndarray): 1D array of index values for coloring the index wedges.
        data_label (str): Data label for colorbar.
        num_regions (int): Number of regions in the plot.
        num_zones (int): Number of zones in each region.
        color_palette (matplotlib.colors.Colormap): Color palette for coloring the wedges.
        start_degree (float): Starting degree of the plot.
        end_degree (float): Ending degree of the plot.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """

    # sort data
    order = np.argsort(-index)
    index = index[order]
    data = data[:,order]
    
    # Calculate start and end angles for each region
    theta1 = np.linspace(start_degree, end_degree, num_regions + 1)
    theta2 = theta1 + (end_degree - start_degree) / num_regions

    patches = []  # Create an empty list of patches

    # Loop over the regions and zones and create wedges
    for i in range(num_regions):
        for j in range(num_zones):
            # Calculate the inner and outer radius for each zone
            r_inner = 1 + j
            r_outer = 2 + j - 0.1

            # Create a wedge with the given parameters
            wedge = Wedge(center=(0, 0), r=r_outer, theta1=theta2[i], theta2=theta1[i], width=r_outer - r_inner,
                          lw=1, facecolor='none', edgecolor='black')

            # Add the wedge to the list of patches
            patches.append(wedge)

    # Create a patch collection from the list of patches
    p = PatchCollection(patches, match_original=True)
    # Set the face colors of the patches according to the data array and color palette
    p.set_array(data.flatten('F'))
    p.set_cmap(color_palette)
    p.set_clim(np.nanpercentile(data[np.nonzero(data)], 10), np.nanpercentile(data, 90))

    # Add the patch collection to the axes
    ax.add_collection(p)
    cax = ax.inset_axes([-5.5, -.75, 4.5, .35], transform=ax.transData)
    colorbar = plt.colorbar(p, ax=ax, cax=cax, shrink=0.5, pad=0.05, orientation='horizontal')
    colorbar.ax.tick_params(labelsize=18, length=0)
    colorbar.ax.set_xlabel(data_label, fontsize=14, rotation=0)

    # Create another list of patches for the index values
    if show_index:
        index_patches = []

        # Loop over the regions and create thin wedges for the index values
        for i in range(num_regions):
            index_wedge = Wedge(center=(0, 0), r=num_zones + 1.5, theta1=theta2[i], theta2=theta1[i], width=0.25)
            index_patches.append(index_wedge)

        # Create another patch collection from the list of patches
        q = PatchCollection(index_patches, match_original=True)
        # Set the face colors of the patches according to the index array and color palette
        q.set_array(index)
        q.set_cmap(color_palette)

        # Add the patch collection to the axes
        ax.add_collection(q)

    # Add zone labels
    zone_labels = ['VZ', 'SVZ', 'IZ', 'SP', 'CP']

    # Loop over the zones and create annotations
    for j in range(num_zones):
        # Calculate the angle of the leftmost segment of each zone
        ang = theta1[0] - (end_degree - start_degree) / num_regions

        r_pos = 0.5 + j
        # Calculate the x and y coordinates of the point on the wedge
        x = np.cos(np.deg2rad(ang)) - (r_pos * 0.8) + 0.2
        y = np.sin(np.deg2rad(ang)) * (r_pos * 1.25) + 0.1

        # Create an annotation
        ax.annotate(zone_labels[j], xy=(x, y), color='black', rotation=ang - 100, ha='right', va='center',
                    fontsize=18)

    # Hide the spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Hide the ticks and tick labels from both axes
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Set the aspect ratio of the axes to equal
    ax.set_aspect('equal')

    # Set the x and y limits of the axes
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
