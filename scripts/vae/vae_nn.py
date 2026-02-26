
import numpy as np

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F





# --- PyTorch VAE module ---

class VAEModule(nn.Module):
	def __init__(self, input_dim, latent_dim, enc_hidden=400, dec_hidden=400):
		super().__init__()
		self.input_dim = int(input_dim)
		self.latent_dim = int(latent_dim)
		# Encoder: input -> enc_hidden -> mu, logvar
		self.enc_fc1 = nn.Linear(input_dim, enc_hidden)
		self.enc_fc_mu = nn.Linear(enc_hidden, latent_dim)
		self.enc_fc_logvar = nn.Linear(enc_hidden, latent_dim)
		# Decoder: z -> dec_hidden -> reconstruction (linear out; MSE on real-valued data)
		self.dec_fc1 = nn.Linear(latent_dim, dec_hidden)
		self.dec_fc2 = nn.Linear(dec_hidden, input_dim)

	def encode(self, x):
		# Returns (mu, logvar) for latent dims.
		h = F.relu(self.enc_fc1(x))
		mu = self.enc_fc_mu(h)
		logvar = self.enc_fc_logvar(h)
		return mu, logvar

	def reparameterize(self, mu, logvar):
		# Reparameterization trick: z = mu + eps * sigma, eps ~ N(0,1).
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def decode(self, z):
		# Latent z -> reconstruction (scaled space).
		h = F.relu(self.dec_fc1(z))
		return self.dec_fc2(h)

	def forward(self, x):
		# Full forward: encode -> reparameterize -> decode; returns (recon, mu, logvar, z).
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		recon = self.decode(z)
		return recon, mu, logvar, z


# --- VAE pipeline: scaler + model, fit / encode / decode ---

class VAEPipeline:
	def __init__(self, n_features, latent_dim, enc_hidden=400, dec_hidden=400,
	             lr=1e-3, max_epochs=50, batch_size=64, device='cpu', beta=1.0):
		# Holds StandardScaler, VAEModule, and training config.
		self.n_features = int(n_features)
		self.latent_dim = int(latent_dim)
		self.enc_hidden = int(enc_hidden)
		self.dec_hidden = int(dec_hidden)
		self.lr = float(lr)
		self.max_epochs = int(max_epochs)
		self.batch_size = int(batch_size)
		self.device = device
		self.beta = float(beta)

		self.x_scaler = StandardScaler()
		self.vae = VAEModule(
			input_dim=n_features,
			latent_dim=latent_dim,
			enc_hidden=enc_hidden,
			dec_hidden=dec_hidden
		).to(device)

	def _loss_fn(self, recon, x, mu, logvar, beta=1.0):
		# Reconstruction (MSE) + beta * KL; KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2).
		MSE = F.mse_loss(recon, x, reduction='sum')
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return MSE + beta * KLD

	def fit(self, X, verbose=True):
		# Fit scaler and train VAE on X (unsupervised).
		X = np.asarray(X, dtype=np.float32)
		if X.ndim == 1:
			X = X.reshape(1, -1)
		Xs = self.x_scaler.fit_transform(X)
		Xt = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
		N = Xt.size(0)
		optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
		self.vae.train()
		last_epoch_loss = None
		for epoch in range(self.max_epochs):
			perm = torch.randperm(N)
			total_loss = 0.0
			n_batches = 0
			for start in range(0, N, self.batch_size):
				end = min(start + self.batch_size, N)
				idx = perm[start:end]
				batch = Xt[idx]
				optimizer.zero_grad()
				recon, mu, logvar, _ = self.vae(batch)
				loss = self._loss_fn(recon, batch, mu, logvar, self.beta)
				loss.backward()
				optimizer.step()
				total_loss += loss.item()
				n_batches += 1
			last_epoch_loss = total_loss / N  # per-sample average
			if verbose and (epoch + 1) % max(1, self.max_epochs // 10) == 0:
				print(f"  VAE epoch {epoch + 1}/{self.max_epochs}  loss/sample = {last_epoch_loss:.4f}")
		self.vae.eval()
		if verbose:
			# Final loss and reconstruction metrics on full data (original scale).
			with torch.no_grad():
				recon_all, mu_all, logvar_all, _ = self.vae(Xt)
				loss_full = self._loss_fn(recon_all, Xt, mu_all, logvar_all, self.beta).item() / N
			recon_np = recon_all.cpu().numpy()
			recon_orig = self.x_scaler.inverse_transform(recon_np)
			mse_orig = np.mean((X - recon_orig) ** 2)
			mae_orig = np.mean(np.abs(X - recon_orig))
			print("  --- VAE training done ---")
			print(f"  samples={N}, features={self.n_features}, latent_dim={self.latent_dim}")
			print(f"  final loss/sample (scaled) = {last_epoch_loss:.4f}")
			print(f"  full-batch loss/sample     = {loss_full:.4f}")
			print(f"  reconstruction MSE (original scale) = {mse_orig:.6f}")
			print(f"  reconstruction MAE (original scale)  = {mae_orig:.6f}")
		return self

	def encode(self, X, return_z=True):
		"""Encode X to latent. return_z=True => sampled z; return_z=False => mu. Returns (mu_np, logvar_np, latent_np); third is z or mu."""
		X = np.asarray(X, dtype=np.float32)
		if X.ndim == 1:
			X = X.reshape(1, -1)
		Xs = self.x_scaler.transform(X)
		Xt = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
		self.vae.eval()
		with torch.no_grad():
			mu, logvar = self.vae.encode(Xt)
			latent = self.vae.reparameterize(mu, logvar) if return_z else mu
		mu_np = mu.cpu().numpy()
		logvar_np = logvar.cpu().numpy()
		latent_np = latent.cpu().numpy()
		return mu_np, logvar_np, latent_np

	def decode(self, z):
		"""z: (batch, latent_dim) or (latent_dim,). Returns reconstruction in scaled space, shape (batch, n_features)."""
		z = np.asarray(z, dtype=np.float32)
		if z.ndim == 1:
			z = z.reshape(1, -1)
		zt = torch.from_numpy(z.astype(np.float32)).to(self.device)
		self.vae.eval()
		with torch.no_grad():
			recon = self.vae.decode(zt)
		return recon.cpu().numpy()

	def decode_from_latent(self, z):
		"""Decode z and inverse-transform to original scale."""
		recon_scaled = self.decode(z)
		return self.x_scaler.inverse_transform(recon_scaled)

	def Encode(self, X, deterministic=True):
		"""Promoted API. Returns latent (1D if single sample). deterministic=True => mu (reproducible); False => sampled z."""
		mu, logvar, latent = self.encode(X, return_z=not deterministic)
		if latent.shape[0] == 1:
			return latent[0]
		return latent

	def Decode(self, z):
		"""Promoted API. z 1D or 2D; returns reconstruction in original scale."""
		return self.decode_from_latent(z)

	def EncodeDecode(self, X, deterministic=True):
		"""Full forward; returns (recon, mu, logvar, latent). deterministic=True => mu used for latent (reproducible)."""
		mu, logvar, latent = self.encode(X, return_z=not deterministic)
		recon = self.decode_from_latent(latent)
		return recon, mu, logvar, latent