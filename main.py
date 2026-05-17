import pandas as pd
from src.WhoIndicators import WhoIndicators
from pathlib import Path
from src.model_optim import grid_search_all_clustering_models
from src.other_models import run_mini_som, run_gmm, run_autoencoder
from pprint import pprint


def download_tidy_clean_preprocess() -> pd.DataFrame:
    who_indicators = WhoIndicators()
    raw_saving_dir = Path("./data/raw")
    who_indicators.download_indicators(raw_saving_dir)
    tidied_saving_dir = Path("./data/tidied/tidied.csv")
    who_indicators.tidy_indicators(tidied_saving_dir)
    cleaned_saving_dir = Path("./data/cleaned/cleaned.csv")
    who_indicators.apply_mice(cleaned_saving_dir)
    preprocessed_saving_dir = Path("./data/preprocessed/preprocessed.csv")
    who_indicators_df = who_indicators.preprocess(preprocessed_saving_dir)
    return who_indicators_df

def main() -> None:
    df = download_tidy_clean_preprocess()
    clustering_models_grid_search_results = grid_search_all_clustering_models(
        df.values, verbose=3
    )
    pprint(clustering_models_grid_search_results)
    clustering_models_grid_search_results.to_csv("./results.csv", index=False)
    mini_som_results = run_mini_som(df.values)
    pprint(mini_som_results)
    gmm_results = run_gmm(df.values)
    pprint(gmm_results)
    autoencoder_results = run_autoencoder(df.values)
    pprint(autoencoder_results)


if __name__ == "__main__":
    main()
