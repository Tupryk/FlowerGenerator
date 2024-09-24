import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

# Hyperparameters
latent_dim = 256  # Size of the noise vector (input to generator)
img_size = 64     # Image size (adjusted for color images)
channels = 3      # Number of channels for color images (RGB)
batch_size = 64   # Batch size
lr = 0.0002       # Learning rate for both networks
epochs = 500       # Number of training epochs
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

flattened_size = 128 * (img_size // 16) * (img_size // 16)


# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, flattened_size),
            nn.Unflatten(1, (128, img_size // 16, img_size // 16)),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        # img = img.view(img.size(0), channels, img_size, img_size)
        return img


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # [batch, 16, 32, 32]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [batch, 32, 16, 16]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [batch, 64, 8, 8]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [batch, 128, 4, 4]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(flattened_size, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )

    def forward(self, img):
        # img_flat = img.view(img.size(0), -1)
        validity = self.model(img)
        return validity

# CIFAR-10 Dataset (or any color image dataset)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Resize images to a uniform size
    transforms.ToTensor(),                     # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Custom Dataset to load all images from multiple folders
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.transform = transform

        # Walk through all subdirectories and collect image paths
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):  # Add more extensions if needed
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Open the image

        if self.transform:
            image = self.transform(image)

        return image

# Define transformations (resize, tensor conversion, normalization, etc.)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Resize images to a uniform size
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize (mean, std) for RGB
])

# Define the root directory where your multiple folders are located
data_dir = './data/flowers'

# Create an instance of the custom dataset
dataset = ImageDataset(root_dir=data_dir, transform=transform)
print("Images in dataset: ", len(dataset))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the Generator and Discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Loss function
adversarial_loss = nn.BCELoss()  # Binary cross entropy

# Training the GAN
for epoch in range(epochs):
    for i, imgs in enumerate(dataloader):
        real_imgs = imgs.to(device)

        # Labels
        real_labels = torch.ones(imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)

        # ==========================
        #  Train Discriminator
        # ==========================
        optimizer_D.zero_grad()

        # Real images
        real_loss = adversarial_loss(discriminator(real_imgs), real_labels)

        # Fake images
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_imgs = generator(z)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake_labels)

        # Total discriminator loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # ==========================
        #  Train Generator
        # ==========================
        optimizer_G.zero_grad()

        # Generate fake images and fool the discriminator
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), real_labels)  # Generator wants D to think fake images are real

        g_loss.backward()
        optimizer_G.step()

        # Print training progress
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader)} "
                  f"Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")

    # Generate and save sample images every few epochs
    if epoch % 10 == 0:
        with torch.no_grad():
            sample_z = torch.randn(64, latent_dim).to(device)
            generated_imgs = generator(sample_z).cpu()
            grid = torchvision.utils.make_grid(generated_imgs, nrow=8, normalize=True)
            plt.imsave(f"ouputs_epoch{epoch}.png", grid.permute(1, 2, 0).numpy())

# After training, generate some final images
with torch.no_grad():
    torch.save(discriminator.state_dict(), f"./models/{discriminator}.pth")
    torch.save(generator.state_dict(), f"./models/{generator}.pth")
    sample_z = torch.randn(64, latent_dim).to(device)
    generated_imgs = generator(sample_z).cpu()
    grid = torchvision.utils.make_grid(generated_imgs, nrow=8, normalize=True)
    plt.imsave(f"final.png", grid.permute(1, 2, 0).numpy())
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title('Generated Images')
    plt.show()
