from load_data import load_data
from model import Encoder, Decoder
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim


latent_dim = 512
mse = nn.MSELoss()

def loss(x, x_pred, mu, sigma, batch_size):
    recon_error = mse(x_pred, x)
    rel_entropy = torch.mean(.5*(torch.sum(torch.exp(sigma) + mu**2 - sigma - 1, dim=1)))
                    #.5*(torch.sum(torch.exp(sigma)) + torch.dot(mu, mu) - latent_dim - torch.sum(sigma))
    return recon_error + 1*rel_entropy

losses = []
Load = True

def train():
    if Load:
        save = torch.load("model.pt")
        encoder.load_state_dict(save["encoder_states"])
        decoder.load_state_dict(save["decoder_states"])

    adam = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=.0001)
    if Load:
        adam.load_state_dict(save["optimizer_states"])
    batch_size = 10
    for epoch in range(0,10):
        for chunk in range(0,20):
            images = load_data(2000,chunk)
            images = images/255 - .5
            for batch in range(0, 2000, batch_size):
                adam.zero_grad()
                epsilon = torch.normal(0, 1, size=(batch_size,latent_dim)).cuda()
                mu, sigma = encoder(images[batch:(batch+batch_size)])

                z = mu + torch.exp(sigma) * epsilon
                pred = decoder(z)
                los = loss(images[batch:(batch+batch_size)] + .5, pred, mu, sigma, batch_size)
                los.backward()
                losses.append(los.item())
                
            # print(los.item())
                """
                print("sigma:")
                print(sigma)
                print("mu:")
                print(mu)
                """
                adam.step()
            print(losses[-1])
        
    torch.save({
        "encoder_states": encoder.state_dict(),
        "decoder_states": decoder.state_dict(),
        "optimizer_states": adam.state_dict(),
        },
        "model.pt")
            



images = load_data(2000,0)
images = images/255-.5


encoder = Encoder(latent_dim).cuda()
decoder = Decoder(latent_dim).cuda()



train()

plt.plot(losses)
plt.show()

mu, sigma = encoder(images[1:2])
print(mu)
print(torch.exp(sigma))

epsilon = torch.normal(0, 1, size=(1,latent_dim)).cuda()
z = mu + torch.exp(sigma) * epsilon

pred = torch.squeeze(decoder(z))
pred = pred.permute(1, 2, 0)
print(pred)
plt.imshow(pred.detach().cpu().numpy())
plt.show()
