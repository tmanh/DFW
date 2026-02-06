import pickle
import random
import matplotlib.pyplot as plt


def draw(edges):
    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))

    for start, end, direction in edges:
        lat1, lon1 = start
        lat2, lon2 = end
        color = [random.random(), random.random(), random.random()]

        # Plot the edge
        ax.plot([lon1, lon2], [lat1, lat2], color=color, linewidth=1.5)

        # Add arrow to indicate direction
        dx, dy = lon2 - lon1, lat2 - lat1
        ax.arrow(
            lon1, lat1, dx * 0.9, dy * 0.9,
            head_width=0.001, head_length=0.0015,
            fc=color, ec=color, length_includes_head=True
        )

    # Customize plot
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Directed Geographic Graph (green=1, red=-1)")
    plt.grid(True)
    plt.tight_layout()

    # Save to image file
    output_path = "directed_graph_map.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


with open('data/data_with_width.pkl', 'rb') as f:
    data_dict = pickle.load(f)

i = 0
for k, v in data_dict.items():
    i += 1

    if i < 10:
        continue
    print(data_dict[k]['sim_graph'].keys())
    # print(data_dict[k]['sim_graph']['nodes'])
    print(data_dict[k]['sim_graph']['edges'])
    draw(data_dict[k]['sim_graph']['edges'])
    exit()
