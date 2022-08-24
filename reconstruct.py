from model import Encoder, Decoder
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
import numpy as np

latent_dim=16

encoder = Encoder(latent_dim).cuda()
decoder = Decoder(latent_dim).cuda()

save = torch.load("model_best.pt")

encoder.load_state_dict(save["encoder_states"])
decoder.load_state_dict(save["decoder_states"])

images = [np.array(Image.open("recon.jpg"))]
images = torch.tensor(np.array(images), dtype=torch.float32).permute(0, 3, 1, 2).cuda()
images = images/255 - .5
print(images.size())

mu, sigma = encoder(images[0:1])
print(mu[0][0:8])
epsilon = torch.normal(0, 0.05, size=(1,latent_dim)).cuda()
z = mu + torch.exp(sigma) * epsilon
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

generated = torch.squeeze(decoder(z)).permute((1,2,0)).detach().cpu().numpy()
gen_img = Image.fromarray((generated*255).astype(np.uint8))
gen_img.save("reconstructed.jpg")