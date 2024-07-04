import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

ngpu = 1
image_size = 64
nc = 3
ndf = 64

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

def is_fake_or_real(discriminator_path, image_path, device):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    netD = Discriminator(ngpu).to(device)
    netD = load_model(netD, discriminator_path, device)
    netD.eval()

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = netD(image).view(-1).item()

    if output > 0.3:
        print(output, image_path, "The image is real.")
    else:
        print(output, image_path, "The image is fake.")

if __name__ == '__main__':
    discriminator_path = "model_25/discriminator.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = [
        "data/input/fake1.jpg", 
        "data/input/fake2.jpg", 
        "data/input/fake3.jpg", 
        "data/input/real1.jpg", 
        "data/input/real2.jpg", 
        "data/input/real3.jpg", 
        "data/input/real4.jpg", 
        "data/input/real5.jpg", 
        "data/input/real6.jpg", 
        "data/input/real7.jpg",
        "data/input/real8.jpg", 
        "data/input/real9.jpg", 
    ]
    for v in input:
        is_fake_or_real(discriminator_path, v, device)
