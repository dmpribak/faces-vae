import torch.nn as nn
from torch.nn.modules import conv
import torch

biggest = 512
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 0)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, (0,1))
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 0)
        self.conv6 = nn.Conv2d(256, 512, 3, 2, 0)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.linear1 = nn.Linear(biggest*2*2, latent_dim)
        self.linear2 = nn.Linear(biggest*2*2, latent_dim)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        # 178x218x3
        conv1 = self.conv1(X)
        conv1 = self.relu(conv1)
        pool1 = self.pool(conv1) #89x109x8
        #print(conv1.size())
        
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        #pool2 = self.pool(conv2) #44x54x16
        #print(conv2.size())
        
        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        #pool3 = self.pool(conv3) #22x27x32
        #print(conv3.size())
        
        conv4 = self.conv4(conv3)
        conv4 = self.relu(conv4)
        #pool4 = self.pool(conv4) #11x13x64
        #print(conv4.size())
        
        conv5 = self.conv5(conv4)
        conv5 = self.relu(conv5)
        #pool5 = self.pool(conv5) #6x5x128
        #print(conv5.size())
        
        conv6 = self.conv6(conv5)
        conv6 = self.relu(conv6)
        #print(conv6.size()) #2x2x512
        
        
        flat = torch.flatten(conv6,1)
        mu = self.linear1(flat)
        sigma = self.linear2(flat)
        
        return mu, sigma
        
        
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        stride = 4
        """
        self.convt1 = nn.ConvTranspose2d(32, 32, 4, 4, 1)
        self.convt2 = nn.ConvTranspose2d(32, 16, 4, 4, (1,2))
        self.convt3 = nn.ConvTranspose2d(16, 8, (3,3), (3,4), (2,2))
        self.convt4 = nn.ConvTranspose2d(8, 8, (3,4), 3, (4,4))
        """
        """
        self.convt1 = nn.ConvTranspose2d(256, 128, 3, (3,3), 0)
        self.convt2 = nn.ConvTranspose2d(128, 64, 3, (3,3), (0,1))
        self.convt3 = nn.ConvTranspose2d(64, 32, (3,2), 3, (0,1))
        self.convt4 = nn.ConvTranspose2d(32, 32, (3,2), 2, 0)
        self.convt5 = nn.ConvTranspose2d(32, 32, (4,2), 2, 1)
        """
        
        self.convt0 = nn.ConvTranspose2d(512, 256, 3, (3,2), 0, (0, 0))
        self.convt1 = nn.ConvTranspose2d(256, 128, 3, 2, 0, 0)
        self.convt2 = nn.ConvTranspose2d(128, 64, 3, 2, (0,1), (0,0))
        self.convt3 = nn.ConvTranspose2d(64, 32, 3, 2, (1,0), 0)
        self.convt4 = nn.ConvTranspose2d(32, 16, 4, 2, 0, 0)
        self.convt5 = nn.ConvTranspose2d(16, 16, 4, 2, 0, 0)
        
        self.conv1 = nn.Conv2d(16, 3, 3, 1, 1)
        
        self.linear1 = nn.Linear(latent_dim, biggest*2*2)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, X):
        linear1 = self.linear1(X)
        linear1 = linear1.view(-1, biggest, 2, 2)
        
        convt0 = self.convt0(linear1)
        convt0 = self.relu(convt0)
        
        convt1 = self.convt1(convt0)
        convt1 = self.relu(convt1)
        
        convt2 = self.convt2(convt1)
        convt2 = self.relu(convt2)
        
        convt3 = self.convt3(convt2)
        convt3 = self.relu(convt3)

        convt4 = self.convt4(convt3)
        convt4 = self.relu(convt4)
        
        convt5 = self.convt5(convt4)
        convt5 = self.relu(convt5)
        
        conv1 = self.conv1(convt5)
        conv1 = self.sigmoid(conv1) 
        
        return conv1
