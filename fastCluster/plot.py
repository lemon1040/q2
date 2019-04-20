import matplotlib.pyplot as plt
import os
savePath = os.getcwd() + r'\data\img' + '\\'


def plot_diagram(x, y, x_label, y_label, title, show_index=False, index=None, above=0):
    styles = ['k', 'g', 'r', 'c', 'm', 'y', 'b', '#9400D3', '#C0FF3E']
    plt.figure(0)
    plt.clf()
    plt.scatter(x, y, s=10, marker='.', color=styles[0])
    if show_index:
        for p_x, p_y, i in zip(x, y, index):
            plt.text(p_x, p_y + above, str(i), ha='center', va='bottom', fontsize=5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(savePath + title + '.png')
    plt.show()


def plot_cluster(x_label, y_label, title, clusters, vectors, boundary=None):
    styles = ['b', 'g', 'r', 'c', 'm', 'y', '#9400D3', '#C0FF3E', '#FFD700', '#FFC0CB']
    plt.figure(0)
    plt.clf()
    for index, cluster in enumerate(clusters):
        for point in cluster:
            plt.plot(vectors[point, 0], vectors[point, 1], marker='.', color=styles[index])
    if boundary is not None:
        for point in boundary:
            plt.plot(vectors[point, 0], vectors[point, 1], marker='.', color='k')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(savePath + title + '.png')
    plt.show()