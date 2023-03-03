import torch
import torch.nn as nn
import torch.nn.functional as F

# class VAE(nn.Module):
#     def __init__(self, input_size, hidden_size, latent_size):
#         super(VAE, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc21 = nn.Linear(hidden_size, latent_size)
#         self.fc22 = nn.Linear(hidden_size, latent_size)
#         self.fc3 = nn.Linear(latent_size, hidden_size)
#         self.fc4 = nn.Linear(hidden_size, input_size)

#         nn.init.kaiming_normal_(self.fc1.weight)
#         nn.init.kaiming_normal_(self.fc21.weight)
#         nn.init.kaiming_normal_(self.fc22.weight)
#         nn.init.kaiming_normal_(self.fc3.weight)
#         nn.init.kaiming_normal_(self.fc4.weight)


#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar


# class VAE(nn.Module):
#     def __init__(self, input_dim=512, latent_dim=32):
#         super(VAE, self).__init__()

#         # Encoder layers
#         self.enc_hidden_1 = nn.Linear(input_dim, 256)
#         self.enc_bn_1 = nn.BatchNorm1d(256)
#         self.enc_activation_1 = nn.LeakyReLU()
#         self.enc_hidden_2 = nn.Linear(256, 128)
#         self.enc_bn_2 = nn.BatchNorm1d(128)
#         self.enc_activation_2 = nn.LeakyReLU()
#         self.enc_hidden_3 = nn.Linear(128, 64)
#         self.enc_bn_3 = nn.BatchNorm1d(64)
#         self.enc_activation_3 = nn.LeakyReLU()
#         # self.enc_hidden_4 = nn.Linear(64, latent_dim)
#         # self.enc_bn_4 = nn.BatchNorm1d(latent_dim)
#         # self.enc_activation_4 = nn.LeakyReLU()

#         self.enc_hidden_4_mu = nn.Linear(64, latent_dim)
#         self.enc_hidden_4_logvar = nn.Linear(64, latent_dim)

#         # Decoder layers
#         self.dec_hidden_1 = nn.Linear(latent_dim, 64)
#         self.dec_bn_1 = nn.BatchNorm1d(64)
#         self.dec_activation_1 = nn.LeakyReLU()
#         self.dec_hidden_2 = nn.Linear(64, 128)
#         self.dec_bn_2 = nn.BatchNorm1d(128)
#         self.dec_activation_2 = nn.LeakyReLU()
#         self.dec_hidden_3 = nn.Linear(128, 256)
#         self.dec_bn_3 = nn.BatchNorm1d(256)
#         self.dec_activation_3 = nn.LeakyReLU()
#         self.dec_hidden_4 = nn.Linear(256, input_dim)

#         self.apply(self.weights_init)


#     def encode(self, x):
#         x = self.enc_hidden_1(x)
#         x = self.enc_bn_1(x)
#         x = self.enc_activation_1(x)
#         x = self.enc_hidden_2(x)
#         x = self.enc_bn_2(x)
#         x = self.enc_activation_2(x)
#         x = self.enc_hidden_3(x)
#         x = self.enc_bn_3(x)
#         x = self.enc_activation_3(x)
#         # x = self.enc_hidden_4(x)
#         # x = self.enc_bn_4(x)
#         # x = self.enc_activation_4(x)
#         mu = self.enc_hidden_4_mu(x)
#         logvar = self.enc_hidden_4_logvar(x)
#         return mu, logvar

#     def decode(self, z):
#         z = self.dec_hidden_1(z)
#         z = self.dec_bn_1(z)
#         z = self.dec_activation_1(z)
#         z = self.dec_hidden_2(z)
#         z = self.dec_bn_2(z)
#         z = self.dec_activation_2(z)
#         z = self.dec_hidden_3(z)
#         z = self.dec_bn_3(z)
#         z = self.dec_activation_3(z)
#         z = self.dec_hidden_4(z)
#         return torch.sigmoid(z)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         z = mu + eps * std
#         return z

#     def forward(self, x):
#         # Encoder
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)

#         # Decoder
#         x_recon = self.decode(z)

#         return x_recon, mu, logvar


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        # Encode input into mean and log variance of the latent space
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)

        # Reparameterize the mean and log variance to get a sample from the latent space
        z = self.reparameterize(mu, logvar)

        # Decode the sample from the latent space
        x_hat = self.decoder(z)

        return x_hat, mu, logvar

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                # xavier_uniform_
                # kaiming_normal_


import torch.nn.functional as F


def tanimoto_loss(x1, x2):
    numerator = torch.sum(torch.mul(x1, x2))
    denominator = torch.sum(torch.add(x1, x2)) - numerator
    return 1.0 - (numerator / denominator)


def loss_fn(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    # recon_loss = nn.MSELoss()(x, recon_x)
    # recon_loss = huber_loss(x, recon_x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # TAN = tanimoto_loss(recon_x, x)
    return recon_loss + KLD  # + TAN


def huber_loss(y, y_pred, delta=1.0):
    error = y - y_pred
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return loss.mean()


import numpy as np
import torch


def split_data(data, split_percent):
    """
    Randomly split a PyTorch tensor by a given percentage into a training set and a validation set.

    Args:
        data (torch.Tensor): the data to be split
        split_percent (float): the percentage of the data to be used for training

    Returns:
        train_data (torch.Tensor): the training set
        val_data (torch.Tensor): the validation set
    """
    # Get the number of data points
    n_data = data.size()[0]

    # Calculate the number of data points to use for training
    n_train = int(n_data * split_percent)

    # Shuffle the data
    shuffled_indices = np.random.permutation(n_data)

    # Split the data into training and validation sets
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:]
    train_data = data[train_indices]
    val_data = data[val_indices]

    return train_data, val_data


def train_vae(x, hidden_size, latent_size, n_epochs):
    x_train, x_val = split_data(x, 0.1)

    train_loader = torch.utils.data.DataLoader(x_train, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(x_val, batch_size=32, shuffle=False)

    # vae = VAE(x.shape[1], hidden_size, latent_size)
    vae = VAE(x.shape[1], latent_size)  # vae2
    vae.double()
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=0.001)

    best_loss = float("inf")
    best_model = None

    # Train the VAE with your data
    for epoch in range(n_epochs):
        for i, (x) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = loss_fn(recon_x, x, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_value_(vae.parameters(), 1)
            optimizer.step()

        # Validate the model on a separate validation set
        with torch.no_grad():
            val_loss = 0.0
            for i, (x_val) in enumerate(val_loader):
                recon_x_val, mu_val, logvar_val = vae(x_val)
                val_loss += loss_fn(recon_x_val, x_val, mu_val, logvar_val)
            val_loss /= len(val_loader)

        # Keep track of the best model and save it
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = vae.state_dict()

        if epoch % 100 == 0:
            print(loss.item())
            print(val_loss, "val loss")

    vae.load_state_dict(best_model)
    torch.save(vae.state_dict(), f"vae-{latent_size}.pth")
    print(f"best loss: {best_loss}")
    return vae


# vae = train_vae(42, 128, 2, 1000)
