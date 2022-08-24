from model import Encoder, Decoder
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
import numpy as np

latent_dim = 512
decoder = Decoder(latent_dim).cuda()

save = torch.load("model.pt")

decoder.load_state_dict(save["decoder_states"])

z = torch.normal(0, .7, (1,latent_dim)).cuda()

generated = torch.squeeze(decoder(z)).permute((1,2,0)).detach().cpu().numpy()

fig,ax = plt.subplots()
plt.subplots_adjust(bottom=.6)


plt.imshow(generated)

sliders=[]
slider_axes=[]
for i in range(0,8):
    slider_axes.append(plt.axes([.25, .5-.05*i, .3, .03]))
    sliders.append(Slider(ax=slider_axes[i], label="", valmin=-3, valmax=3, valinit=z[0][i].detach().cpu()))
    
for i in range(0,8):
    slider_axes.append(plt.axes([.6, .5-.05*i, .3, .03]))
    sliders.append(Slider(ax=slider_axes[i+8], label="", valmin=-3, valmax=3, valinit=z[0][i+8].detach().cpu()))


def update(val):
    for i in range(0,16):
        z[0][i] = sliders[i].val
    generated = torch.squeeze(decoder(z)).permute((1,2,0)).detach().cpu().numpy()
    ax.imshow(generated)
    

for slider in sliders:
    slider.on_changed(update)
    
plt.show()

gen_img = Image.fromarray((generated*255).astype(np.uint8))
gen_img.save("generated.jpg")