import torch
import numpy as np
from models import Diffuser, LATENT_DIM

IMAGE_DIMS = 64
DIFFUSION_STEPS = 256
np_data = np.load("./data/arrays_latent.npz")["arr_0"]
device = "cpu"
dataset = torch.utils.data.TensorDataset(torch.Tensor(np_data))
loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)


def get_alpha_betas(N: int):
    """Schedule from the original paper."""
    beta_min = .1
    beta_max = 20.
    betas = np.array([beta_min/N + i/(N*(N-1))*(beta_max-beta_min) for i in range(N)])
    alpha_bars = np.cumprod(1 - betas)
    return alpha_bars, betas


def train(nepochs: int=1_000, denoising_steps: int=DIFFUSION_STEPS, save_as: str="./models/diffuser.pth"):
    """Alg 1 from the DDPM paper"""
    model = Diffuser()
    model.load_state_dict(torch.load('./models/diffuser.pth'))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    alpha_bars, _ = get_alpha_betas(denoising_steps)      # Precompute alphas
    losses = []
    for epoch in range(nepochs):
        for [data] in loader:
            data = data.to(device)
            optimizer.zero_grad()

            # Fwd pass
            t = torch.randint(denoising_steps, size=(data.shape[0],))  # sample timesteps - 1 per datapoint
            alpha_t = torch.index_select(torch.Tensor(alpha_bars), 0, t).unsqueeze(1).to(device)    # Get the alphas for each timestep
            noise = torch.randn(*data.shape, device=device)   # Sample DIFFERENT random noise for each datapoint
            model_in = alpha_t**.5 * data + noise*(1-alpha_t)**.5   # Noise corrupt the data (eq14)
            out = model(model_in, t.unsqueeze(1).to(device))
            loss = torch.mean((noise - out)**2)     # Compute loss on prediction (eq14)
            losses.append(loss.detach().cpu().numpy())

            # Bwd pass
            loss.backward()
            optimizer.step()

        if (epoch+1) % 1 == 0:
            mean_loss = np.mean(np.array(losses))
            losses = []
            print("Epoch %d,\t Loss %f " % (epoch+1, mean_loss))

    torch.save(model.state_dict(), save_as)
    return model


def sample(model, n_samples: int=50, n_steps: int=DIFFUSION_STEPS):
    """Alg 2 from the DDPM paper."""
    x_t = torch.randn((n_samples, LATENT_DIM)).to(device)
    alpha_bars, betas = get_alpha_betas(n_steps)
    alphas = 1 - betas
    for t in range(len(alphas))[::-1]:
        ts = t * torch.ones((n_samples, 1)).to(device)
        ab_t = alpha_bars[t] * torch.ones((n_samples, 1)).to(device)  # Tile the alpha to the number of samples
        z = (torch.randn((n_samples, LATENT_DIM)) if t > 1 else torch.zeros((n_samples, LATENT_DIM))).to(device)
        model_prediction = model(x_t, ts)
        x_t = 1 / alphas[t]**.5 * (x_t - betas[t]/(1-ab_t)**.5 * model_prediction)
        x_t += betas[t]**0.5 * z

    return x_t

if __name__ == "__main__":
    trained_model = train(100)

    samples = sample(trained_model, 1).detach().cpu().numpy()
    print(samples)
