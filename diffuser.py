import torch
import numpy as np
from models import Diffuser, LATENT_DIM
from diffusers import DDPMScheduler

IMAGE_DIMS = 64
DIFFUSION_STEPS = 512
np_data = np.load("./data/arrays_latent.npz")["arr_0"]
device = "cpu"
dataset = torch.utils.data.TensorDataset(torch.Tensor(np_data))
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)


def train(nepochs: int=1_000, denoising_steps: int=DIFFUSION_STEPS, save_as: str="./models/diffuser.pth"):
    """Alg 1 from the DDPM paper"""
    model = Diffuser()
    model.load_state_dict(torch.load('./models/diffuser_epoch1000.pth'))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = DDPMScheduler(denoising_steps)
    losses = []
    for epoch in range(nepochs):
        for [data] in loader:
            data = data.to(device)
            optimizer.zero_grad()

            # Fwd pass
            noise = torch.randn(*data.shape, device=device)
            t = torch.randint(low=0, high=denoising_steps, size=(1,))
            model_in = scheduler.add_noise(data, noise, torch.IntTensor([DIFFUSION_STEPS-1]))
            t = torch.ones(model_in.shape[0], 1) * t
            out = model(model_in, t)
            loss = torch.mean((data - out)**2)
            losses.append(loss.detach().cpu().numpy())

            # Bwd pass
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            mean_loss = np.mean(np.array(losses))
            losses = []
            print("Epoch %d,\t Loss %f " % (epoch+1, mean_loss))
            if (epoch+1) % 250 == 0:
                torch.save(model.state_dict(), f"./models/diffuser_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), save_as)
    return model


def sample(model, n_samples: int=50, n_steps: int=DIFFUSION_STEPS):
    x = torch.randn((n_samples, LATENT_DIM)).to(device)
    scheduler = DDPMScheduler(n_steps)
    with torch.no_grad():
        for t in reversed(range(n_steps)):
            timestep = torch.ones(n_samples, 1) * t
            predicted_denoised_image = model(x, timestep)
            beta_t = scheduler.betas[t].to(device)
            alpha_t_bar = scheduler.alphas_cumprod[t].to(device)

            x = predicted_denoised_image * alpha_t_bar.sqrt() + torch.randn_like(x) * beta_t.sqrt() if t > 0 else predicted_denoised_image
            print("timestamp", t, x)

    return x

if __name__ == "__main__":
    trained_model = train(1000)

    samples = sample(trained_model, 1).detach().cpu().numpy()
    print(samples)
