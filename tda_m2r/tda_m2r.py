def plot_circle(ax, center, radius, **kwargs):
    import matplotlib.pyplot as plt
    circle = plt.Circle(center, radius, **kwargs)
    ax.add_artist(circle)


def vrips(point_cloud, r, circle=False, ax=None, p_scale=1):
    from scipy.spatial.distance import pdist, squareform
    from itertools import combinations
    import matplotlib.pyplot as plt
    import numpy as np

    if point_cloud.shape[1] != 2:
        raise NotImplementedError("Point cloud is not two-dimensional.")

    distances = squareform(pdist(point_cloud))

    # Threshold for creating edges and simplices (epsilon)

    # Create a plot
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Set the size and color of the points
    # Adjust this value for larger or smaller points
    point_size = 25 / np.sqrt(p_scale)
    point_color = 'black'

    # Scatter plot with adjusted point size and color
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1],
               s=point_size, c=point_color, zorder=3)

    # Plot circles around points
    if circle:
        for point in point_cloud:
            plot_circle(ax, point, r, edgecolor='r', facecolor='none',
                        linestyle='dotted', zorder=1, alpha=0.5)

    # Plot edges (1-simplices)
    for i in range(len(point_cloud)):
        for j in range(i + 1, len(point_cloud)):
            if distances[i, j] < r:
                ax.plot([point_cloud[i, 0], point_cloud[j, 0]],
                        [point_cloud[i, 1], point_cloud[j, 1]], 'k-',
                        alpha=0.5, zorder=2)

    # Plot 2-simplices (triangles)
    for i, j, k in combinations(range(len(point_cloud)), 3):
        if distances[i, j] < r and distances[i, k] < r and distances[j, k] < r:
            triangle = np.array(
                [point_cloud[i], point_cloud[j], point_cloud[k]])
            ax.fill(triangle[:, 0], triangle[:, 1],
                    'maroon', alpha=0.3, zorder=0)

    if not ax:
        plt.title(f"Vietoris-Rips Complex, $\\varepsilon = {r}$")
        plt.xlabel("X")
        plt.ylabel("Y")
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_title(f"Vietoris-Rips Complex, $\\varepsilon = {r}$")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box')

    if not ax:
        plt.show()


def cech(point_cloud, r, circle=True, ax=None, p_scale=1):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist, squareform
    from itertools import combinations

    if point_cloud.shape[1] != 2:
        raise NotImplementedError("Point cloud is not two dimensional.")

    # Step 2: Compute pairwise distances
    distances = squareform(pdist(point_cloud))

    # Create a plot
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(point_cloud[:, 0], point_cloud[:, 1],
               s=25 / np.sqrt(p_scale), zorder=3, c='black')

    # Plot circles (0-simplices)

    if circle:
        for point in point_cloud:
            plot_circle(ax, point, r, edgecolor='b', facecolor='none',
                        linestyle='dotted', zorder=1, alpha=0.5)

    # Plot edges (1-simplices)
    for i in range(len(point_cloud)):
        for j in range(i + 1, len(point_cloud)):
            if distances[i, j] < 2 * r:
                ax.plot([point_cloud[i, 0], point_cloud[j, 0]],
                        [point_cloud[i, 1], point_cloud[j, 1]], 'k-',
                        alpha=0.5, zorder=2)

    # Plot 2-simplices (filled triangles)
    for i, j, k in combinations(range(len(point_cloud)), 3):
        if (distances[i, j] < 2 * r) and (distances[i, k] < 2 * r) and (
                distances[j, k] < 2 * r):
            triangle = np.array(
                [point_cloud[i], point_cloud[j], point_cloud[k]])
            ax.fill(triangle[:, 0], triangle[:, 1], 'b', alpha=0.3, zorder=0)

    if not ax:
        plt.title("Čech Complex, "+f"$\\varepsilon = {r}$")
        plt.xlabel("X")
        plt.ylabel("Y")
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_title(f"Čech Complex, $\\varepsilon = {r}$")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box')

    if not ax:
        plt.show()


def diagram_sizes(dgms):
    return ", ".join([f"$|H_{i}|$={len(d)}" for i, d in enumerate(dgms)])


def gd_barcode(dgm, axes=None, dim=None, fontsize=16):
    import gudhi
    import matplotlib.pyplot as plt
    import numpy as np

    """
    Takes in list of arrays of birth and death.
    Arrays should be arranged such that the first element belongs to H0, and
    so on.
    """

    barcode = []
    if dim == None:
        for dim, bars in enumerate(dgm):
            new_arr = np.insert(bars, 0, dim, axis=1)
            for i in new_arr:
                barcode.append((int(i[0]), tuple(i[1:])))
    else:
        if not isinstance(dgm, np.ndarray):
            raise TypeError(f"Not an array of barcode for dimension {dim}")

        for i in dgm:
            barcode.append((dim, tuple(i)))

    if not axes:
        gudhi.plot_persistence_barcode(barcode, legend=True, fontsize=fontsize)
        plt.show()
    else:
        gudhi.plot_persistence_barcode(
            barcode, legend=True, axes=axes, fontsize=fontsize)


def gd_persistence(dgm, dim=None, axes=None, fontsize=16, greyblock=False):
    import gudhi
    import matplotlib.pyplot as plt
    import numpy as np

    """
    Takes in list of arrays of birth and death.
    Arrays should be arranged such that the first element belongs to H0, and
    so on.
    """

    barcode = []
    if dim == None:
        for d, bars in enumerate(dgm):
            new_arr = np.insert(bars, 0, d, axis=1)
            for i in new_arr:
                barcode.append((int(i[0]), tuple(i[1:])))
    else:
        if not isinstance(dgm, np.ndarray):
            raise TypeError(f"Not an array of barcode for dimension {dim}")

        for i in dgm:
            barcode.append((dim, tuple(i)))

    if not axes:
        gudhi.plot_persistence_diagram(
            barcode, legend=True, fontsize=fontsize, greyblock=greyblock)
        plt.show()
    else:
        gudhi.plot_persistence_diagram(
            barcode, legend=True, axes=axes, fontsize=fontsize, greyblock=greyblock)
