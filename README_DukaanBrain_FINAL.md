# DukaanBrain

DukaanBrain is a small machine-learning project I built to explore how
retail transaction data can be turned into useful business insights for
small and medium-sized stores.

The idea is simple: a shop may already have months of billing data, but
raw transaction rows are difficult to act on. DukaanBrain tries to
convert that history into answers to questions such as:

-   Which products and categories contribute the most to sales?
-   Who are the most valuable customers?
-   Which customers are becoming inactive?
-   Can basic historical patterns provide a signal about daily sales?

This repository currently contains the first working prototype. The
strongest ML component is customer segmentation using RFM features and
K-Means clustering. I also built a simple Linear Regression model as a
baseline daily-sales prediction experiment.

## Dataset

The project uses the **Retail Store Sales Transactions** dataset from
Kaggle.

The data contains transaction-level information such as:

-   Date
-   Customer ID
-   Transaction ID
-   SKU category
-   SKU/product identifier
-   Quantity
-   Sales amount

I also derive an approximate unit price as:

`UnitPrice = Sales_Amount / Quantity`

## What I did

### 1. Data cleaning and understanding

I first inspected the shape, data types, missing values, duplicates and
number of unique customers, transactions, products and categories.

The date column was converted to a proper datetime type and calendar
features such as month, day and day of week were extracted.

### 2. Exploratory data analysis

The EDA focuses on business-oriented questions rather than only plotting
variables.

I looked at:

-   the distribution of sales amounts;
-   highest-demand SKUs;
-   highest-revenue categories;
-   monthly sales patterns;
-   highest-spending customers;
-   transaction value and purchased quantity.

The sales distribution is strongly right-skewed. I did not automatically
delete large observations because inspecting them showed that high sales
values could correspond to plausible quantity and unit-price
combinations.

### 3. Customer segmentation with RFM

For each customer I created three behavioural features:

-   **Recency** --- days since the customer's most recent purchase;
-   **Frequency** --- number of distinct transactions made by the
    customer;
-   **Monetary** --- total amount spent by the customer.

Because K-Means is distance-based and these variables have very
different numerical scales, I standardised the RFM features using
`StandardScaler`.

I then used K-Means to create four **business-oriented exploratory
segments** and interpreted their average RFM behaviour.

In the current run, the four profiles are approximately:

  -----------------------------------------------------------------------------
  Segment              Recency       Frequency        Monetary Interpretation
  ------------ --------------- --------------- --------------- ----------------
  Inactive              273.85            1.70           35.31 Purchased long
                                                               ago, low
                                                               frequency and
                                                               low spending

  Occasional             69.01            2.25           41.75 Relatively
                                                               recent but
                                                               infrequent,
                                                               low-value
                                                               customers

  Loyal                  57.87           10.55          307.57 Repeat customers
                                                               with much
                                                               stronger
                                                               engagement

  VIP                    21.93           25.04         1106.74 Very recent,
                                                               frequent and
                                                               high-value
                                                               customers
  -----------------------------------------------------------------------------

The cluster IDs themselves do not have business meaning. I assigned the
names only after examining the RFM profile of each cluster.

I also compared silhouette scores for several values of `k`. The highest
score in the current experiment occurs at `k=2`, so the four-cluster
solution should **not** be described as the mathematically optimal
clustering. I kept four clusters as an exploratory business segmentation
because it produces more actionable customer profiles. A future version
should validate this choice more thoroughly.

### 4. Baseline daily-sales prediction

I aggregated transaction rows into daily sales and created simple
calendar features:

-   day;
-   month;
-   day of week.

I then trained a Linear Regression model using a train/test split.

Current baseline results:

-   **MAE:** about 1124.80
-   **RMSE:** about 1396.72
-   **R²:** about 0.283

An R² of about 0.28 does **not** mean 28% accuracy. It means that, under
this evaluation setup, the model explains about 28% of the variation in
the test-set daily sales.

This model is intentionally simple and should be treated as a **baseline
prediction experiment**, not a production time-series forecasting
system. The current random split is convenient for an initial regression
benchmark but does not preserve chronological order.

## Why two different ML approaches?

The customer problem and sales problem are different.

For customer segmentation, the dataset has no labels such as "VIP" or
"Inactive", so I use **unsupervised learning (K-Means)** to discover
groups with similar behaviour.

For sales prediction, daily sales provide a numerical target, so I use
**supervised regression** and evaluate predictions against known test
values.

## Tech stack

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Scikit-learn
-   Jupyter Notebook

## Project structure

``` text
DukaanBrain/
├── data/
│   └── scanner_data.csv
├── notebooks/
│   └── 01_data_exploration.ipynb
├── README.md
└── requirements.txt
```

## Running the notebook

Install the main dependencies:

``` bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

Then open the notebook and run the cells from top to bottom.

## What I learned

The most useful part of this project was seeing how raw transaction data
has to be transformed before ML becomes meaningful. For example, K-Means
is not run directly on transaction rows. I first had to decide what
customer behaviour means, create RFM features, scale them, inspect the
resulting groups and only then translate those clusters into
business-friendly segments.

The baseline regression experiment also showed that building a model is
not the same as building a strong predictor. Simple calendar features
capture only part of daily sales behaviour, which gives a clear
direction for improving the project.

## Limitations

This is an early prototype.

Some important limitations are:

-   the dataset covers roughly one year, so long-term seasonality cannot
    be established confidently;
-   the dataset does not contain promotions, holidays, inventory
    availability, product costs or profit margins;
-   K-Means can be sensitive to skewed variables and outliers;
-   four clusters are used for business interpretability even though the
    current silhouette comparison favours two clusters;
-   the Linear Regression model uses only basic calendar features;
-   the current regression train/test split is random, so I describe it
    as a baseline prediction experiment rather than rigorous future
    forecasting.

## Next steps

The next versions of DukaanBrain could include:

-   lagged sales and rolling-average features;
-   chronological/time-series validation;
-   stronger regression models such as Random Forest or gradient-boosted
    trees;
-   product-level demand prediction;
-   market-basket analysis for products frequently purchased together;
-   a Streamlit dashboard so a store owner can use the insights without
    opening a notebook.

## Project goal

The long-term goal of DukaanBrain is not to give a shopkeeper
complicated ML outputs. It is to translate retail data into simple
actions: identify valuable customers, understand demand, spot customers
who may be drifting away and make better inventory and sales decisions.
