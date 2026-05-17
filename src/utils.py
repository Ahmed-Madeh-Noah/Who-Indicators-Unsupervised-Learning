import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from src.scorer import dbcv_scorer
import joblib
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from minisom import MiniSom
from sklearn.mixture import GaussianMixture


class FullDataSplit:
    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        yield indices, indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1


def get_n_clusters(estimator, best_params, X):
    estimator.set_params(**best_params)
    estimator.fit(X)
    n_clusters = np.unique(estimator.labels_).shape[0]
    n_clusters = n_clusters - 1 if -1 in np.unique(estimator.labels_) else n_clusters
    return n_clusters


def test_estimator(estimator: type, X, **estimator_params):
    estimator = estimator(**estimator_params)
    estimator.fit(X)
    labels = estimator.labels_
    silhouette = silhouette_score(X, labels)
    dbcv = dbcv_scorer(estimator, X)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    return {
        "Silhouette Score": silhouette,
        "DBVC Score": dbcv,
        "Calinski Harabasz Score": calinski_harabasz,
        "Davies Bouldin Score": davies_bouldin,
    }


def generate_cluster_trajectory_fig(model, country):
    model_path = f"./models/{model}.pkl"
    model = joblib.load(model_path)
    df = pd.read_csv("./data/preprocessed/preprocessed.csv")
    df["Org_Year"] = pd.read_csv(
        "./data/cleaned/cleaned.csv", usecols=lambda col: col == "Year"
    )["Year"]
    df_work = df.copy()
    features = df_work.drop(columns=["Country", "Org_Year"])
    if isinstance(model, dict):
        encoder = model["AutoEncoder"]
        kmeans = model["KMeans"]
        encoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(features.values)
            _, latent_features = encoder(X_tensor)
            latent_features_np = latent_features.numpy()
        cluster_assignments = kmeans.predict(latent_features_np)
        df_work["Cluster"] = cluster_assignments
    elif isinstance(model, MiniSom):
        winners = [model.winner(row) for row in features.values]
        df_work["Cluster"] = [f"{w[0]}_{w[1]}" for w in winners]
    elif isinstance(model, GaussianMixture):
        df_work["Cluster"] = model.predict(features.values)
    else:
        model.fit(features)
        df_work["Cluster"] = model.labels_
    tsne_coords = TSNE(n_components=2, random_state=42).fit_transform(features)
    df_work[["TSNE1", "TSNE2"]] = tsne_coords
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=df_work,
        x="TSNE1",
        y="TSNE2",
        hue="Cluster",
        palette="viridis",
        alpha=0.4,
        s=50,
        ax=ax,
        legend="full",
    )
    country_data = df_work[df_work["Country"] == country].sort_values(by="Org_Year")

    if not country_data.empty:
        ax.plot(
            country_data["TSNE1"],
            country_data["TSNE2"],
            color="crimson",
            linewidth=2.5,
            marker="o",
            markersize=8,
            label=f"{country} Trajectory",
            zorder=5,
        )
        n_points = len(country_data)
        for i, (_, row) in enumerate(country_data.iterrows()):
            if i == 0 or i == n_points - 1:
                ax.annotate(
                    str(int(row["Org_Year"])),
                    (row["TSNE1"], row["TSNE2"]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                    weight="bold",
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="white", alpha=0.9, ec="none"
                    ),
                )

    ax.set_title(
        f"Cluster Landscape and {country}'s Trajectory Over Time", fontsize=14, pad=15
    )
    ax.set_xlabel("TSNE1", fontsize=11)
    ax.set_ylabel("TSNE2", fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    return fig
