from pathlib import Path
import requests
from tqdm import tqdm
import pandas as pd
import re
from hyperimpute.plugins.imputers import Imputers
from sklearn.preprocessing import StandardScaler
from pprint import pprint


class WhoIndicators:
    API_ENDPOINT = "https://data.who.int/api/datadot/ddiindicatorperspectives"
    UNNECESSARY_COLUMNS = [
        "IND_ID",
        "DIM_GEO_CODE_M49",
        "IND_UUID",
        "DIM_PUBLISH_STATE_CODE",
        "IND_PER_CODE",
        "DIM_TIME_TYPE",
        "DIM_GEO_CODE_TYPE",
        "IND_CODE",
    ]
    MUTUAL_COLUMNS = ["Country", "Year"]

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.indicator_name_path_dict = dict()

    def download_indicators(self, saving_dir: Path) -> dict:
        self.raw_saving_dir = saving_dir
        api_response = requests.get(WhoIndicators.API_ENDPOINT)
        if api_response.status_code != 200:
            if self.verbose:
                print(f"Could not reach the API endpoint {WhoIndicators.API_ENDPOINT}")
            return self.indicator_name_path_dict
        indicators = api_response.json()
        progress_bar = tqdm(indicators.get("value", []), desc="Downloading Indicators")
        for indicator in progress_bar:
            indicator_data_export_uri = indicator.get("DataExportUri", "")
            if not indicator_data_export_uri:
                continue
            indicator_response = requests.get(indicator_data_export_uri)
            if indicator_response.status_code != 200:
                continue
            indicator_name = Path(indicator_data_export_uri).name
            indicator_file_path = self.raw_saving_dir / indicator_name
            with open(indicator_file_path, "wb") as indicator_file:
                indicator_file.write(indicator_response.content)
            self.indicator_name_path_dict[indicator_name] = indicator_file_path
        return self.indicator_name_path_dict

    def tidy_indicators(self, saving_path: Path) -> pd.DataFrame:
        self.tidied_saving_path = saving_path
        csv_files = tuple(self.raw_saving_dir.glob("*.csv"))
        progress_bar = tqdm(csv_files, desc="Tidying Indicators")

        def select_necessary_columns(col: str) -> bool:
            return col not in WhoIndicators.UNNECESSARY_COLUMNS

        mut_cols = WhoIndicators.MUTUAL_COLUMNS
        indicator_df_list = list()
        for csv_file in progress_bar:
            indicator_df = pd.read_csv(csv_file, usecols=select_necessary_columns)
            new_cols = {col: col.replace("DIM_", "") for col in indicator_df.columns}
            indicator_df = indicator_df.rename(columns=new_cols)
            indicator_name = re.sub(r"\W", "", indicator_df["IND_NAME"].iloc[0].title())
            indicator_df = indicator_df.drop(columns=["IND_NAME"])
            new_feature_names = {
                feature: f"{indicator_name}-{re.sub(r'\W', '', feature.replace('_', ' ').title())}"
                for feature in indicator_df.columns
                if feature not in ("GEO_NAME_SHORT", "TIME")
            }
            new_feature_names["GEO_NAME_SHORT"] = "Country"
            new_feature_names["TIME"] = "Year"
            indicator_df = indicator_df.rename(columns=new_feature_names)
            indicator_df = indicator_df.drop_duplicates(subset=mut_cols)
            indicator_df_list.append(indicator_df)
        tidied_df = indicator_df_list[0]
        progress_bar = tqdm(indicator_df_list[1:], desc="Merging Indicators")
        for indicator_df in progress_bar:
            tidied_df = pd.merge(tidied_df, indicator_df, on=mut_cols, how="outer")
        tidied_df = tidied_df.dropna(subset=mut_cols)
        tidied_df = tidied_df.sort_values(by=mut_cols, ascending=[True, True])
        tidied_df.reset_index(drop=True)
        self.tidied_df = tidied_df
        tidied_df.to_csv(self.tidied_saving_path, index=False)
        return tidied_df

    def apply_mice(self, saving_path: Path) -> pd.DataFrame:
        self.cleaned_saving_path = saving_path
        mostly_non_nan_mask = self.tidied_df.isnull().mean() < 0.5
        mostly_non_nan_cols = self.tidied_df.columns[mostly_non_nan_mask]
        self.cleaned_df = self.tidied_df[mostly_non_nan_cols]
        countries_df = self.cleaned_df[["Country"]]
        num_cols = self.cleaned_df.select_dtypes(exclude=str).columns
        num_cols_df = self.cleaned_df[num_cols]
        self.mice = Imputers().get("mice")
        cleaned_df = self.mice.fit_transform(num_cols_df)
        cleaned_df.columns = num_cols_df.columns
        self.cleaned_df = pd.merge(
            countries_df, cleaned_df, left_index=True, right_index=True
        )
        self.cleaned_df.to_csv(self.cleaned_saving_path, index=False)
        return self.cleaned_df

    def preprocess(self, saving_path: Path) -> pd.DataFrame:
        self.preprocessed_saving_path = saving_path
        num_cols = self.cleaned_df.select_dtypes(exclude=str).columns
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(self.cleaned_df[num_cols])
        self.preprocessed_df = pd.DataFrame(
            scaled,
            columns=self.scaler.get_feature_names_out(),
        )
        self.preprocessed_df["Country"] = self.cleaned_df["Country"]
        self.preprocessed_df.to_csv(self.preprocessed_saving_path, index=False)
        return self.preprocessed_df


if __name__ == "__main__":
    who_indicators = WhoIndicators(verbose=True)
    raw_saving_dir = Path("./data/raw")
    indicator_name_path_dict = who_indicators.download_indicators(raw_saving_dir)
    pprint(indicator_name_path_dict)
    tidied_saving_path = Path("./data/tidied/tidied.csv")
    who_indicators_df = who_indicators.tidy_indicators(tidied_saving_path)
    pprint(who_indicators_df.head())
    pprint(who_indicators_df.info(verbose=True))
    pprint(who_indicators_df.describe(include="all"))
    cleaned_saving_path = Path("./data/cleaned/cleaned.csv")
    who_indicators_df = who_indicators.apply_mice(cleaned_saving_path)
    pprint(who_indicators_df.head())
    pprint(who_indicators_df.info(verbose=True))
    pprint(who_indicators_df.describe(include="all"))
    preprocessed_saving_path = Path("./data/preprocessed/preprocessed.csv")
    who_indicators_df = who_indicators.preprocess(preprocessed_saving_path)
    pprint(who_indicators_df.head())
    pprint(who_indicators_df.info(verbose=True))
    pprint(who_indicators_df.describe(include="all"))
