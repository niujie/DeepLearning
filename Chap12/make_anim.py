import imageio
import glob
from IPython import display
import os

directory = './vae_images'
os.chdir(directory)

anim_file = 'vae-gen.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('sampled_epoch*.png')
    filenames = sorted(filenames)
    filenames.sort(key=lambda x: int(x[13:-4]))  # 文件名按数字排序
    last = -1
    for i, filename in enumerate(filenames):
        image = imageio.imread(filename)
        writer.append_data(image)

import IPython

if IPython.version_info > (6, 2, 0, ''):
    display.Image(filename=anim_file)
