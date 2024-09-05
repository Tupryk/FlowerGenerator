import torch
import numpy as np
import matplotlib.pyplot as plt
from models import Autoencoder, Diffuser
from diffuser import sample

IMAGE_DIMS = 64

def generate_flower():
    diffuser = Diffuser()
    diffuser.load_state_dict(torch.load('./models/diffuser_epoch1000.pth'))
    diffuser.eval()
    latent_im = sample(diffuser, n_samples=4*4, n_steps=512).detach().cpu().numpy()
    latent_im = latent_im.reshape(4*4, 128, 4, 4)
    latent_im = torch.Tensor(latent_im)

    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load('./models/autoencoder.pth'))
    autoencoder.eval()

    images = autoencoder.decoder(latent_im)
    images = torch.nan_to_num(images, nan=0.0)

    images = images.detach().cpu().numpy()
    images = np.maximum(images, 0)
    images = np.minimum(images, 1)
    return images

if __name__ == "__main__":
    # Create a figure with 4x4 subplots
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    # Generate and plot each image
    images = generate_flower()
    for i in range(4):
        for j in range(4):
            image = np.transpose(images[i*4+j], (1, 2, 0))
            axes[i, j].imshow(image)
            axes[i, j].axis('off')  # Hide the axis
    
    plt.tight_layout()
    plt.show()

