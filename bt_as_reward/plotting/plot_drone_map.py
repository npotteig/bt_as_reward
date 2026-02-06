import os

import numpy as np

from bt_as_reward.path_planning.convert_binvox import ConvertBinvox
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Times New Roman"]})
rc("text", usetex=False)
plt.rcParams["font.size"] = 24

ALTITUDE = 5.0
CENTERING_OFFSET = np.array([-49.5, -49.5])

if __name__ == "__main__":
    cmap = ListedColormap(["lightgray", "blue"])

    np.random.seed(0)
    # parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cb = ConvertBinvox()
    obs_map, width, height, _ = cb.to_map(
        cb.load_binvox(os.path.join("data", "voxelgrid_400_400_50.binvox"))
    )
    obs_map_reshaped = obs_map.reshape((width, height))
    # 1634 obstacle voxel grids
    rr_obs_map = cb.get_slice(obs_map_reshaped, altitude=ALTITUDE, tolerance=0)[
        188:212, 188:212
    ]
    indices = np.column_stack(np.where(rr_obs_map == 1))
    # new_indices = filter_interior_points(indices)
    new_indices = indices
    new_map = np.zeros_like(rr_obs_map)
    for x, y in new_indices:
        new_map[x, y] = 1

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("x(m)")
    ax.set_ylabel("y(m)")
    ax.imshow(new_map, cmap=cmap, alpha=0.5, extent=[-12, 12, -12, 12])

    rows, cols = 3, 2
    cell_width = 24 / cols
    cell_height = 24 / rows

    # Draw rectangles and labels
    label = 1
    for row in reversed(range(rows)):
        for col in range(cols):
            x = (col * cell_width) - 12
            y = (row * cell_height) - 12

            # Draw rectangle
            rect = patches.Rectangle(
                (x, y),
                cell_width,
                cell_height,
                linewidth=0.5,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

            # Add label in the center of the cell
            ax.text(
                x + cell_width / 2,
                y + cell_height / 2,
                str(label),
                color="black",
                fontsize=20,
                fontweight="bold",
                ha="center",
                va="center",
            )
            label += 1
    # rect = patches.Rectangle(
    #     (30, 15), 1, 2,
    #     linewidth=1,
    #     edgecolor='green',
    #     facecolor='green',
    # )
    # ax.add_patch(rect)

    # rect = patches.Rectangle(
    #     (0, 0), 1, 1,
    #     linewidth=1,
    #     edgecolor='black',
    #     facecolor='black',
    # )
    # rotation = transforms.Affine2D().rotate_deg_around(0.5, 0.5, np.degrees(-45))

    # rect.set_transform(rotation + ax.transData)

    # triangle = patches.Polygon(
    #             [[0.5, 0.5], [4.5, 10], [-3.5, 10]],
    #             edgecolor='white',
    #             facecolor='white',
    #             alpha=1.0
    #         )
    # triangle.set_transform(rotation + ax.transData)
    # ax.add_patch(triangle)

    # ax.add_patch(rect)
    plt.savefig("plots/drone_map.pdf")
    np.save("data/drone_obstacles.npy", new_indices)
