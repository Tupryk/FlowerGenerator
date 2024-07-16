import torch
import numpy as np
import matplotlib.pyplot as plt
from models import Autoencoder

IMAGE_DIMS = 64
np_data = np.load("./data/arrays.npz")["arr_0"]
device = "cpu"
dataset = torch.utils.data.TensorDataset(torch.Tensor(np_data))
loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)


def train(nepochs: int = 10, save_as: str="./models/autoencoder.pth"):
    model = Autoencoder()
    model.load_state_dict(torch.load('./models/autoencoder.pth'))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for epoch in range(nepochs):
        for [data] in loader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data)

            loss = torch.mean((data - out)**2)
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

if __name__ == "__main__":
    model = train(100)
    # model = Autoencoder()
    # model.load_state_dict(torch.load('./models/autoencoder.pth'))
    model.eval()

    # See results
    image = np_data[np.random.randint(0, len(np_data))]
    y = model.forward(torch.Tensor(image))

    image = image.reshape(IMAGE_DIMS, IMAGE_DIMS, 3)
    y = y.detach().cpu().numpy()
    y = (y + 1.) * .5
    y = y.reshape(IMAGE_DIMS, IMAGE_DIMS, 3)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].axis('off')
    ax[1].imshow(y)
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()