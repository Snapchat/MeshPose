import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from meshpose.utils.meshpose_inference import MeshPoseInference
from meshpose.utils import imread

CMAP = mcolors.LinearSegmentedColormap.from_list('RedGreen', ['red', 'green'])


def showverts(image, verts, size, color):
    plt.imshow(image)
    for axis in [0, 2]:
        plt.scatter(verts[:, axis], verts[:, 1], s=size, color=color)
    plt.show()


if __name__ == '__main__':

    image_path = 'assets/pexels-olly-3799115.jpg'
    bbox = [584, 32, 556, 704]  # COCO bbox definition (x,y,w,h)

    image = imread(image_path)

    meshpose = MeshPoseInference()
    outputs = meshpose(image, bbox)

    showverts(image, outputs['xyz_hp'], size=1, color=None)
    showverts(image, outputs['xyz_lp'], size=5, color=CMAP(outputs['vertex_vis_lp']))
