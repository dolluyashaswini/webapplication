import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        self.encoder=nn.Sequential(
                  nn.Linear(28*28,256),
                  nn.ReLU(True),
                  nn.Linear(256,128),
                  nn.ReLU(True),
                  nn.Linear(128,64),
                  nn.ReLU(True)
        
                  )
    
        self.decoder=nn.Sequential(
                  nn.Linear(64,128),
                  nn.ReLU(True),
                  nn.Linear(128,256),
                  nn.ReLU(True),
                  nn.Linear(256,28*28),
                  nn.Sigmoid(),
                  )
    
 
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
    
        return x
def load_model():
	model=autoencoder()
	model.load_state_dict(torch.load("models/MNISTdenoisingweights.pth",map_location=torch.device('cpu')))
	return model
def load_transforms():
	return transforms.Compose([
           transforms.Resize(32),
           transforms.CenterCrop(28),   
           transforms.Grayscale(num_output_channels=1),
           transforms.ToTensor(),
           transforms.Normalize(mean=0,std=1)])
def predict(net,transform,image):
    img=transform(image).unsqueeze(0)
    img=img.view(img.size(0),-1)
    output=net(img)
    output=output.view(output.size(0),1,28,28)
    return output
