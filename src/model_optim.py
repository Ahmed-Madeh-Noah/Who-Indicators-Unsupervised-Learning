from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    BisectingKMeans,
    DBSCAN,
    OPTICS,
    AgglomerativeClustering,
)
from sklearn.model_selection import GridSearchCV
import joblib
from src.utils import get_n_clusters
from src.utils import FullDataSplit
from src.scorer import (
    silhouette_scorer,
    dbcv_scorer,
    calinski_harabasz_scorer,
    neg_davies_bouldin_scorer,
)
import pandas as pd
from pathlib import Path
from pprint import pprint


def grid_search_kmeans(X, scorers, refit_scorer, split, verbose):
    if verbose:
        print("\nGrid Searching KMeans")
    kmeans = KMeans(random_state=42)
    param_grid = {"n_clusters": list(range(4, 11))}
    grid_search = GridSearchCV(
        estimator=kmeans,
        param_grid=param_grid,
        scoring=scorers,
        n_jobs=1,
        refit=refit_scorer,
        cv=split,
        verbose=verbose,
    )
    grid_search.fit(X)
    joblib.dump(
        kmeans.set_params(**grid_search.best_params_).fit(X), "./models/KMeans.pkl"
    )
    return {
        "N_Clusters": grid_search.best_params_["n_clusters"],
        "Score": grid_search.best_score_,
        "Best_Parameters": grid_search.best_params_,
    }


def grid_search_mini_batch_kmeans(X, scorers, refit_scorer, split, verbose):
    if verbose:
        print("\nGrid Searching MiniBatchKMeans")
    mini_batch_kmeans = MiniBatchKMeans(random_state=42)
    param_grid = {
        "n_clusters": list(range(4, 11)),
        "batch_size": [10, 100, 1000],
    }
    grid_search = GridSearchCV(
        estimator=mini_batch_kmeans,
        param_grid=param_grid,
        scoring=scorers,
        n_jobs=1,
        refit=refit_scorer,
        cv=split,
        verbose=verbose,
    )
    grid_search.fit(X)
    joblib.dump(
        mini_batch_kmeans.set_params(**grid_search.best_params_).fit(X), "./models/MiniBatchKMeans.pkl"
    )
    return {
        "N_Clusters": grid_search.best_params_["n_clusters"],
        "Score": grid_search.best_score_,
        "Best_Parameters": grid_search.best_params_,
    }


def grid_search_bisecting_kmeans(X, scorers, refit_scorer, split, verbose):
    if verbose:
        print("\nGrid Searching BisectingKMeans")
    bisecting_kmeans = BisectingKMeans(random_state=42)
    param_grid = {
        "n_clusters": list(range(4, 11)),
        "bisecting_strategy": ["biggest_inertia", "largest_cluster"],
    }
    grid_search = GridSearchCV(
        estimator=bisecting_kmeans,
        param_grid=param_grid,
        scoring=scorers,
        n_jobs=1,
        refit=refit_scorer,
        cv=split,
        verbose=verbose,
    )
    grid_search.fit(X)
    joblib.dump(
        bisecting_kmeans.set_params(**grid_search.best_params_).fit(X), "./models/BisectingKMeans.pkl"
    )
    return {
        "N_Clusters": grid_search.best_params_["n_clusters"],
        "Score": grid_search.best_score_,
        "Best_Parameters": grid_search.best_params_,
    }


def grid_search_dbscan(X, scorers, refit_scorer, split, verbose):
    if verbose:
        print("\nGrid Searching DBSCAN")
    dbscan = DBSCAN(n_jobs=1)
    param_grid = {
        "eps": [0.05, 0.1, 0.2, 0.4],
        "min_samples": [5, 10, 20, 40, 80],
    }
    grid_search = GridSearchCV(
        estimator=dbscan,
        param_grid=param_grid,
        scoring=scorers,
        n_jobs=1,
        refit=refit_scorer,
        cv=split,
        verbose=verbose,
    )
    grid_search.fit(X)
    joblib.dump(
        dbscan.set_params(**grid_search.best_params_).fit(X), "./models/DBSCAN.pkl"
    )
    return {
        "N_Clusters": get_n_clusters(dbscan, grid_search.best_params_, X),
        "Score": grid_search.best_score_,
        "Best_Parameters": grid_search.best_params_,
    }


def grid_search_optics(X, scorers, refit_scorer, split, verbose):
    if verbose:
        print("\nGrid Searching OPTICS")
    optics = OPTICS(n_jobs=1)
    param_grid = {
        "min_samples": [5, 10, 20, 40, 80],
        "min_cluster_size": [5, 20, 80],
    }
    grid_search = GridSearchCV(
        estimator=optics,
        param_grid=param_grid,
        scoring=scorers,
        n_jobs=1,
        refit=refit_scorer,
        cv=split,
        verbose=verbose,
    )
    grid_search.fit(X)
    joblib.dump(
        optics.set_params(**grid_search.best_params_).fit(X), "./models/OPTICS.pkl"
    )
    return {
        "N_Clusters": get_n_clusters(optics, grid_search.best_params_, X),
        "Score": grid_search.best_score_,
        "Best_Parameters": grid_search.best_params_,
    }


def grid_search_agglomerative_clustering(X, scorers, refit_scorer, split, verbose):
    if verbose:
        print("\nGrid Searching AgglomerativeClustering")
    agglomerative_clustering = AgglomerativeClustering()
    param_grid = {
        "n_clusters": list(range(4, 11)),
        "linkage": ["ward", "complete", "average", "single"],
    }
    grid_search = GridSearchCV(
        estimator=agglomerative_clustering,
        param_grid=param_grid,
        scoring=scorers,
        n_jobs=1,
        refit=refit_scorer,
        cv=split,
        verbose=verbose,
    )
    grid_search.fit(X)
    joblib.dump(
        agglomerative_clustering.set_params(**grid_search.best_params_).fit(X), "./models/AgglomerativeClustering.pkl"
    )
    return {
        "N_Clusters": grid_search.best_params_["n_clusters"],
        "Score": grid_search.best_score_,
        "Best_Parameters": grid_search.best_params_,
    }


def grid_search_all_clustering_models(X, verbose=0):
    clustering_models_grid_search_results_list = list()
    scorers = {
        "silhouette_scorer": silhouette_scorer,
        "dbcv_scorer": dbcv_scorer,
        "calinski_harabasz_scorer": calinski_harabasz_scorer,
        "neg_davies_bouldin_scorer": neg_davies_bouldin_scorer,
    }
    centroid_refit_scorer = "silhouette_scorer"
    density_refit_scorer = "dbcv_scorer"
    hierarchical_refit_scorer = "neg_davies_bouldin_scorer"
    split = FullDataSplit()

    kmeans_grid_search_results_dict = grid_search_kmeans(
        X, scorers, centroid_refit_scorer, split, verbose
    )
    clustering_models_grid_search_results_list.append({
        "Model": "KMeans",
        "Scorer": centroid_refit_scorer,
        **kmeans_grid_search_results_dict,
    })

    mini_batch_kmeans_grid_search_results_dict = grid_search_mini_batch_kmeans(
        X, scorers, centroid_refit_scorer, split, verbose
    )
    clustering_models_grid_search_results_list.append({
        "Model": "MiniBatchKMeans",
        "Scorer": centroid_refit_scorer,
        **mini_batch_kmeans_grid_search_results_dict,
    })

    bisecting_kmeans_grid_search_results_dict = grid_search_bisecting_kmeans(
        X, scorers, centroid_refit_scorer, split, verbose
    )
    clustering_models_grid_search_results_list.append({
        "Model": "BisectingKMeans",
        "Scorer": centroid_refit_scorer,
        **bisecting_kmeans_grid_search_results_dict,
    })

    dbscan_grid_search_results_dict = grid_search_dbscan(
        X, scorers, density_refit_scorer, split, verbose
    )
    clustering_models_grid_search_results_list.append({
        "Model": "DBSCAN",
        "Scorer": density_refit_scorer,
        **dbscan_grid_search_results_dict,
    })

    optics_grid_search_results_dict = grid_search_optics(
        X, scorers, density_refit_scorer, split, verbose
    )
    clustering_models_grid_search_results_list.append({
        "Model": "OPTICS",
        "Scorer": density_refit_scorer,
        **optics_grid_search_results_dict,
    })

    agglomerative_clustering_grid_search_results_dict = (
        grid_search_agglomerative_clustering(
            X, scorers, hierarchical_refit_scorer, split, verbose
        )
    )
    clustering_models_grid_search_results_list.append({
        "Model": "AgglomerativeClustering",
        "Scorer": hierarchical_refit_scorer,
        **agglomerative_clustering_grid_search_results_dict,
    })

    return pd.DataFrame(clustering_models_grid_search_results_list)


if __name__ == "__main__":
    preprocessed_saving_path = Path("./data/preprocessed/preprocessed.csv")
    df = pd.read_csv(preprocessed_saving_path).drop(columns="Country")
    clustering_models_grid_search_results_path = Path(
        "./data/metadata/clustering_models_grid_search_results.csv"
    )
    results_df = grid_search_all_clustering_models(df.values, verbose=3)
    results_df.to_csv(clustering_models_grid_search_results_path, index=False)
    pprint(results_df)
