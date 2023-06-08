import networkx as nx 
from skimage import io

import matplotlib.pyplot as plt

def plot_closest_imgs(anc_img_names, DATA_DIR, image, img_path, closest_idx, distance, no_of_closest=10):
    G = nx.Graph()
    S_name = [img_path.split('/')[-1]]
    closest_matches = []  # List to store closest image names

    for s in range(no_of_closest):
        S_name.append(anc_img_names.iloc[closest_idx[s]])
        closest_matches.append(anc_img_names.iloc[closest_idx[s]])  # Store closest image names

    for i in range(len(S_name)):
        image = io.imread(DATA_DIR + S_name[i])
        G.add_node(i, image=image)

    for j in range(1, no_of_closest + 1):
        G.add_edge(0, j, weight=distance[closest_idx[j - 1]])

    pos = nx.kamada_kawai_layout(G)

    # Rest of the plotting code...

    return closest_matches
