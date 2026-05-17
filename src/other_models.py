from minisom import MiniSom
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


def run_mini_som(X):
    n_range = range(4, 11)
    tes = []
    print("MiniSom")
    for n in n_range:
        mini_som = MiniSom(x=n, y=n, input_len=X.shape[1], random_seed=42)
        mini_som.random_weights_init(X)
        mini_som.train(X, X.shape[0])
        tes.append(mini_som.topographic_error(X))
        print(
            f"n={n} topographic_error={mini_som.topographic_error(X)} quantization_error={mini_som.quantization_error(X)}"
        )
    optimal_te_n = list(n_range)[np.argmin(tes)]
    n = optimal_te_n
    mini_som = MiniSom(x=n, y=n, input_len=X.shape[1], random_seed=42)
    mini_som.random_weights_init(X)
    mini_som.train(X, X.shape[0])
    joblib.dump(mini_som, "./models/MiniSom.pkl")
    return {
        "Quantization_Error": mini_som.quantization_error(X),
        "Topographic_Error": mini_som.topographic_error(X),
    }


def run_gmm(X):
    n_range = range(4, 11)
    bics = []
    print("GuassianMixture")
    for n in n_range:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        print(f"n={n} AIC={gmm.aic(X)} BIC={gmm.bic(X)} score={gmm.score(X)}")
    optimal_bic_n = list(n_range)[np.argmin(bics)]
    n = optimal_bic_n
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X)
    joblib.dump(gmm, "./models/GaussianMixture.pkl")
    return {
        "AIC": gmm.aic(X),
        "BIC": gmm.bic(X),
        "Score": gmm.score(X),
    }


class MinimalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(MinimalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def run_autoencoder(X):
    X_tensor = torch.FloatTensor(X)
    input_dim = X_tensor.shape[1]
    latent_dim = 2

    model = MinimalAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        reconstructed, _ = model(X_tensor)
        loss = criterion(reconstructed, X_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        _, latent_features = model(X_tensor)
        latent_features_np = latent_features.numpy()

    silhouette_scores = []
    n_range = range(4, 11)
    print("AutoEncoder_KMeans")
    for num_clusters in n_range:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(latent_features_np)
        silhouette_scores.append(
            silhouette_score(latent_features_np, cluster_assignments)
        )
        print(f"n_clusters={num_clusters} silhouette_scorer={silhouette_scores[-1]}")
    max_silhouette_score = np.max(silhouette_scores)
    optimal_n = list(n_range)[np.argmax(silhouette_scores)]
    kmeans = KMeans(n_clusters=optimal_n, random_state=42)
    kmeans.fit(latent_features_np)
    joblib.dump(
        {"AutoEncoder": model, "KMeans": kmeans}, "./models/AutoEncoder_KMeans.pkl"
    )
    return {
        "n_clusters": optimal_n,
        "silhouette_score": max_silhouette_score,
    }
