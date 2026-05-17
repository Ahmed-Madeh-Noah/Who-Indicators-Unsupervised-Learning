# Who-Indicators-Unsupervised-Learning
Unsupervised machine learning to analyze World Health Organization (WHO) socio-economic and health indicators. Using clustering techniques (like K-Means) and PCA, it categorizes countries to help NGOs strategically identify regions in direst need of humanitarian aid.


## Data Collection
This project fetches public health indicator datasets directly from the World Health Organization (WHO) API.

### How It Works
The `WhoIndicators` class connects to the WHO data insights endpoint, iterates through all available indicator metadata, and downloads the underlying data exports locally.

1. **API Discovery:** Queries the primary WHO Indicator Perspectives API endpoint.
2. **Batch Downloading:** Parses individual `DataExportUri` targets and downloads them sequentially with a progress bar (`tqdm`).
3. **Storage:** Saves the raw files to `data/raw/`.


## Data Tidying
The `WhoIndicators` class includes a built-in automated tidying pipeline (`tidy_indicators`) designed to transform unstructured, isolated indicator CSV files from the World Health Organization (WHO) into a unified, clean, and analysis-ready tabular dataset.

The pipeline addresses common data quality issues such as redundant metadata, inconsistent column naming conventions, and duplicate observations.

### How It Works

The tidying process performs the following operations sequentially on all downloaded CSV files:

1. **Column Filtering**
    * Drops redundant administrative keys and system UUIDs (`IND_ID`, `DIM_GEO_CODE_M49`, `IND_UUID`, etc.) during data loading using a dynamic predicate (`usecols`).

2. **Standardization & Prefixes**
    * Removes the metadata prefix `DIM_` from all remaining structural columns.
    * Standardizes core tracking indices: `GEO_NAME_SHORT` becomes **`Country`** and `TIME` becomes **`Year`**.

3. **Feature Dynamic Renaming**
    * Extracts the unique indicator title from the `IND_NAME` column.
    * Standardizes the title into `PascalCase` (removing all non-alphanumeric characters).
    * Prepends this standardized indicator name to its respective metrics (e.g., `VALUE` becomes `LifeExpectancyAtBirth-Value`). This prevents column name collisions during dataset merging.

4. **Deduplication**
    * Removes duplicate records across the unique composite key entity (`Country`, `Year`) within each individual indicator file.

5. **Outer Joining & Alignment**
* Merges all processed indicator datasets iteratively using a full `outer` join on `['Country', 'Year']`. This ensures no historical data points are lost, even if an indicator is missing for specific countries or timeframes.
    * Drops any resulting artifacts that lack proper country or year identifiers.
    * Sorts the final dataset chronologically and alphabetically by `Country` and `Year`.

6. **Storage**
    * Loads the raw data from `./data/raw/*.csv`.
    * Exports tided data to `./data/tidied/tidied.csv`.


## Data Cleaning
To ensure the dataset is robust enough for unsupervised clustering algorithms (like K-Means and PCA), which cannot inherently handle missing entries, the `WhoIndicators` class includes an automated missing data imputation pipeline (`apply_mice`). 

This step transitions the dataset from a sparse, tidied format into a fully populated, modeling-ready matrix.

### How It Works
The cleaning process utilizes advanced state-of-the-art imputation techniques to handle missing values without discarding critical country profiles.

1. **Multivariate Imputation by Chained Equations (MICE)**
    * Instead of using naive statistical metrics like the mean or median—which collapse variance and distort relationships between features—the pipeline implements the **MICE** algorithm via the `hyperimpute` library.
    * **MICE** models each feature with missing values as a function of all other features in the dataset in an iterative, chained series of predictive models. This preserves the multi-dimensional correlations inherent in socio-economic and health indicators.

2. **Full Matrix Resolution**
    * The predictive models iteratively scan and fill missing cells across the entire merged `tidied.csv` dataframe. 
    * This ensures that developing nations or years with sparse reporting are still accurately represented in the final data pool based on their overall multi-variable trajectory.

3. **Storage**
    * Loads the data directly from the in-memory tidied pipeline or `./data/tidied/tidied.csv`.
    * Exports the fully imputed dataset to `./data/cleaned/cleaned.csv`.


## Data Preprocessing
Before feeding the data into distance-based clustering algorithms (like K-Means) and dimensionality reduction techniques (like PCA), the `WhoIndicators` class executes a structured preprocessing pipeline (`preprocess`). This step ensures that categorical variables are numerically encoded and that all numeric features are on an identical scale, preventing variables with large magnitudes from dominating the models.

### How It Works
The preprocessing pipeline separates columns by data type, transforms them using robust encoding and scaling techniques, and recombines them into a single modeling-ready dataset.

1. **Categorical Feature Encoding**
    * Automatically identifies categorical columns (e.g., text-based features) using a data-type selector.
    * Implements **Binary Encoding** via `BinaryEncoder`. Compared to standard One-Hot Encoding, Binary Encoding converts categories into binary code and splits the digits into separate columns. This drastically reduces dimensionality while maintaining a clean, non-sparse numerical representation of categorical relationships.

2. **Numeric Feature Scaling**
    * Dynamically isolates all numeric columns in the cleaned dataset.
    * Applies a **StandardScaler** to normalize the features. This scales the data so that it has a mean of $0$ and a standard deviation of $1$ ($z$-score normalization), which is a vital prerequisite for stable K-Means clustering and accurate PCA variance mapping.

3. **Feature Alignment & Reconstruction**
    * Concatenates the encoded categorical matrix and the scaled numeric matrix along the column axis (`axis=1`).
    * Reconstructs a clean `pd.DataFrame` by extracting the post-transformation column names directly from the fitted transformer metadata via `.get_feature_names_out()`.

4. **Storage**
    * Loads the fully imputed data directly from `./data/cleaned/cleaned.csv`.
    * Exports the final, preprocessed dataset to the specified saving directory (typically `./data/preprocessed/preprocessed.csv`).
