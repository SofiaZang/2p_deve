from skimage import io
import animatplot as amp
import matplotlib.pyplot as plt

video_ds = io.imread('C:\\Users\\zaggila\\Documents\\pixelNMF\\data\\sz92_2024-06-06_a_cell_control\\AVG_cam_crop_5x_avg.tif') # initialise tiff

block = amp.blocks.Imshow(video_ds)
anim = amp.Animation([block])

anim.controls()
# anim.save_gif('ising')
plt.show()