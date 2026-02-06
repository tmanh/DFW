import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_pyramid_3d(save_path="pyramid_3d.png"):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")

    # pyramid levels (bottom → top)
    sizes = [1, 2, 3, 4]   # width/depth
    heights = [0, 1, 2, 3] # z offsets
    colors = ["#FFD93D", "#FFC300", "#FFB000", "#E68A00"]

    for size, z, color in zip(sizes, heights, colors):
        # corners of cube
        x = [-size/2, size/2, size/2, -size/2]
        y = [-size/2, -size/2, size/2, size/2]

        # 8 corners of the cuboid
        verts = [
            # bottom
            list(zip(x, y, [z]*4)),
            # top
            list(zip(x, y, [z+1]*4)),
            # sides
            [(x[0], y[0], z), (x[1], y[1], z), (x[1], y[1], z+1), (x[0], y[0], z+1)],
            [(x[1], y[1], z), (x[2], y[2], z), (x[2], y[2], z+1), (x[1], y[1], z+1)],
            [(x[2], y[2], z), (x[3], y[3], z), (x[3], y[3], z+1), (x[2], y[2], z+1)],
            [(x[3], y[3], z), (x[0], y[0], z), (x[0], y[0], z+1), (x[3], y[3], z+1)],
        ]

        ax.add_collection3d(Poly3DCollection(
            verts, facecolors=color, edgecolors="k", linewidths=1, alpha=0.9
        ))

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0, 5)
    ax.view_init(elev=20, azim=35)  # camera angle
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

draw_pyramid_3d()
