# Toxicstew: a cookbook on how to go viral on Reddit

# Find our [Datastory here](https://galgantar.github.io/standard-deviants/) 
## This is the [Repository for the Datastory](https://github.com/galgantar/standard-deviants)

## Abstract

Sometimes becoming viral happens by accident — a single post can transform an unknown user into a household name, launch careers, or even make you a Millionarie. Reddit, with its 430+ million monthly active users, has become a breeding ground for viral content that shapes public discourse, from GameStop’s stock surge to political movements and cultural phenomena. We can see that there is plenty of reasons for wanting to author a viral post but how does one do that? The first thing you should probably do is talk about something interesting and add your original view of the matter - in the contex of Reddit this means posting a so-called cross-linking post in which you refer to another post and add your own context to it. But just posting any corss-linking post will not be enough, a good viral post mixes a number of imprtant characteristics to stand out from hundreads thousands of posts posted each day. To help you make sense of all this we assembled <ins>Toxic Stew</ins> - a cookbook containing all the ingredients you need to cook up the perfect cross-linking post and become the next Reddit sensation.

## Research questions

- **Primary**: Which combinations of linguistic, temporal, and network factors have the most evidence of leading to success on Reddit?
- **Sentiment Analysis**: How do sentiment intensity, emotional toxicity, writing styles and aggressive communication patterns correlate with engagement metrics?
- **Network Effects**: What role do subreddit crosslinking patterns and community interactions play in content propagation? How do crosslinking posts within or across clusters of similar subreddits (by users) perform compared to others?
- **Temporal Dynamics**: What is the optimal timing to post or crosslink for maximum visibility and interactions across different subreddit communities?
- **Toxicity-Virality Relationship**: Is there a systematic relationship between controversial content and viral success, and what are the ethical implications? Do supportive or hate comments engage more people?
- **Community-Specific Patterns**: How do viral patterns differ across subreddit communities with varying toxicity levels?

## Proposed Additional Datasets

### Primary Dataset: Reddit Hyperlinks Dataset
- **Source**: [Stanford SNAP Reddit Hyperlinks Dataset](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)
- **Size**: 860k hyperlinks between 51k subreddits
- **Format**: TSV files with source/target subreddits, timestamps, link sentiments and post metadata
- **Preprocessing**: NetworkX MultiDiGraph -> aggregated sentiment DiGraph

### Enrichment Dataset: Reddit Embeddings
- **Source**: [Stanford SNAP Reddit Embeddings](https://snap.stanford.edu/data/web-RedditEmbeddings.html)
- **Size**: 56k subreddit embeddings (300-dimensional vectors)
- **Format**: CSV with subreddit names and embedding vectors
- **Preprocessing**: Merge with primary dataset, perform clustering

### Reddit API Data
- **Source**: Reddit API scraping of all 860k posts from hyperlinks dataset
- **Size**: Metadata of 860k posts obtained from Reddit API
- **Format**: Key metrics obtained 
    - `ups`: Upvotes (primary virality indicator)
    - `downs`: Downvotes (controversy measure)
    - `num_comments`: Comment count (engagement depth)
    - `score`: Net score (ups - downs)
    - `upvote_ratio`: Upvote percentage
- **Preprocessing**: Merging existing data with the primary dataset
- **Note**: A small percetage Reddit posts in the dataset cannot be found, therefore the data is missing for those. Presumably they have been deleted, so an additional analysis can show which reddit posts are likely to get deleted.

## Data handling pipelines

Raw data is uploaded to `/data` folder, the data preprocessing can be found in `/src/data`, enhancing of the data is in `/src/data/scraping`, clustering of embeddings happens in `/src/analysis/subreddit_clustering`. Further analysis can be found in `/src/analysis` and functions which provide basic functions for further analysis are in `/src/utils`.

> **Note**: If you want to fully reproduce the entire pipeline in `results.ipynb`, see the [How to use](#how-to-use) section below.

## Methods

### Regression

#### Linear Regression
- **Purpose**: Fit a linear model to the data.
- **Formula**: We find best $\beta_0, \ldots, \beta_K$ such that the mean squared error (MSE) of individual errors $\epsilon_i$ is the smallest possible with $Y_i = \beta_0 + \beta_1X_{i1} + ... + \beta_KX_{iK} + \epsilon_i$
- **Implementation**: `smf.ols()`
- **Application in dataset**: Predict similarity between subreddits, number of posts overtime and post virality. We can also use weights of the linear regression for explainability and analyzing relationships between features and outcomes.

#### Logistic Regression
- **Purpose**: Fit a logistic regression model to the data to predict the probability for the datapoint corresponding to class 1.
- **Formula**: Find best $\beta_0, \ldots, \beta_K$ such that the $P(Y_i=1|X_i) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_{i1} + ... + \beta_K X_{ik})}}$ minimizes binary cross entropy loss.
- **Implementation**: `smf.logit()`
- **Application in dataset**: Predict post sentiment, interpret relationships between features. Logistic regression is also a key ingredient in calculating the propensity score for matching.


### Trees
#### Decision Trees
- **Purpose**: Fit a decision tree to the data
- **Formula**: Binary tree which splits the data again repeatedly. Chosses the feature and splitting value with the highest information gain at each split
- **Implementation**: `sklearn.tree.DecisionTreeClassifier`
- **Application in dataset**: Predicting virality, introducing game which traverses Decision Tree

#### Random Forests
- **Purpose**: Fitting a number of trees together to the data
- **Formula**: Group of binary trees which work together with majority votes and randomness to get better results
- **Implementation**: `sklearn.tree.RandomForestClassifier`
- **Application in dataset**: Predicting virality, finding feature importances

### Clustering

#### Leiden Clustering
- **Purpose**: Unsupervised algorithm for detecting communities among the vectors.
- **Formula**: Partition $G = (V, E)$ to maximize modularity $Q = \frac{1}{2m} \sum_{i,j} \bigg( A_{i,j} - \frac{k_ik_j}{2m} \bigg) \delta(c_i,c_j)$
- **Implementation**: `igraph.community_leiden()`
- **Application in dataset**: Identify clusters of subreddits with similar user bases and interaction patterns.

### Correlation Coefficients

#### Pearson Correlation Coefficient
- **Purpose**: Measure the strength and direction of the linear relationship between two continuous variables.
- **Formula**: $r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2} \sqrt{\sum_i (y_i - \bar{y})^2}}$ where $r \in [-1,1]$ with 1 = perfect positive linear correlation, -1 = perfect negative, 0 = no linear correlation
- **Implementation**: `scipy.stats.pearsonr()`
- **Application in dataset**: Calculating covariance between LIWC features among subreddits, calculating pairwise covariance between subreddits' LIWC profiles

#### Spearman Correlation Coefficient
- **Purpose**: Assess the monotonic relationship between two ranked variables; captures nonlinear but monotonic associations.
- **Formula**: $\rho = 1 - \frac{6\sum_i d_i^2}{n(n^2 - 1)}$ where $d_i$ is the rank difference for observation $i$, and $\rho \in [-1,1]$
- **Implementation**: `scipy.stats.spearmanr()`
- **Application in dataset**: Studying correlation between toxicity and virality, engagement and LIWC features, and temporal analysis.

### Hypothesis Testing

#### t-Test
- **Purpose**: Test whether the means of two groups are significantly different.
- **Formula**: (independent samples) $t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$ where $\bar{X}_1, \bar{X}_2$ are sample means, $s_1,s_2$ are standard deviations, and $n_1,n_2$ are sample sizes
- **Implementation**: `scipy.stats.ttest_ind()`
- **Application in dataset**: Determining whether posts are viral with statistical significance, whether controversy/negativity drives engagement.

#### KS Test for Normality
- **Purpose**: Test whether a sample follows a specified continuous distribution (commonly the normal distribution).
- **Formula**: $D = \sup_x \ | F_n(x) - F(x) |$ where $F_n(x)$ is the empirical cumulative distribution function (ECDF) and $F(x)$ is the theoretical CDF.
- **Implementation**: `scipy.stats.kstest()`
- **Application in dataset**: Comparing engagement / toxicity distributions between clusters

### Matching

#### Propensity Score Matching
- **Purpose**: Create balanced treatment and control groups by matching units with similar probabilities of receiving treatment, reducing selection bias in observational studies.
- **Formula**: 
  1. Estimate propensity scores: $e(X_i) = P(T_i=1|X_i)$ using logistic regression
  2. Match treated units to control units with similar $e(X)$ using max weight matching in graphs
  3. Estimate treatment effect: $\tau = \frac{1}{n_1}\sum_{T_i=1} (Y_i - Y_{j(i)})$ where $j(i)$ is the matched control for treated unit $i$
- **Implementation**: `smf.logit()`, `nx.max_weight_matching()`
- **Application in dataset**: Comparing virality of toxic vs. non-toxic posts with similar baseline characteristics (LIWC features, subreddit, timing), isolate causal effect of toxicity on engagement by controlling for confounders (e.g. post length, readability, timestamp).


## Proposed timeline

**October 2025**
- Proposals, literature review
- Exploratory data analysis (distributions, missing values)
- Data loading (`NetworkX`, `Pandas`)
- Graph data preprocessing (`MultiDiGraph` with all posts -> `DiGraph` with aggregated sentiments)
- Dataset enrichment with Reddit API
- Dataset enrichment wiht embeddings and clustering of subreddits 
- Top 10 most active subreddits analysis
- Temporal analysis or negative sentiment and upvotes
- Finalising the `results.ipynb` notebook

**November 2025**

- Experiment with different indicators of virality (`ups`, `downs`, `num_comments`)
- Extensive statistical analysis (t-tests) for the impact of different properties on the upvotes
- In depth analysis of the effect timing
- Per cluster analysis - are there any patterens to virality that are only expressed in certain communities 
- Hypothesis tests for all the proposed actions for more informative browsing
- Matching to verify effect of certain features on the upvotes
- Website creation
- Extract the handbook of rules for virality from our experiments

**December 2025**
- Defining Metrics for Virality
- Doing textual analysis
- Matching to verify effect of certain features on the upvotes
- Website creation
- Finalization of the experiments
- Interactive Flourish visualizations for the experiments
- Adding minigames
- Polishing story
- Finishing up with all the experiments
- Polishing website


## Internal milestones
- [x] Graph Data preprocessing (Gal)
- [x] Data loading (Gal, Henrik)
- [x] Enriching data by scraping reddit (Henrik, Gal)
- [x] Temporal analysis (Aziz)
- [x] Clustering of embeddings (Jack)
- [x] Top 10 most active subreddits analysis with hypothesis testing (Jane)
- [x] Sentiment graph structure analysis (Gal)
- [x] Analyze correlations in the data with pearsonr and methods we learned at the tutorials
- [x] Further t-tests and analyses
- [x] Matching to test the effect on the number of upvotes (Jack)
- [x] Per cluster correlations analysis (analyze if there are some cluster specific virality patterens)


## How to use

Install packages

```bash
pip install -r pip_requirements.txt
```

If you don't use vscode you have to export `PYTHONPATH` environment variable to the root of the project for the relative imports to work
```bash
export PYTHONPATH=$(pwd)
```

### Download original datasets

Download files `soc-redditHyperlinks-body.tsv` and `soc-redditHyperlinks-title.tsv` form [Stanford SNAP Reddit Hyperlinks Dataset](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) and put them into `/data`. Then also download the file `web-redditEmbeddings-subreddits.csv` from [Stanford SNAP Reddit Embeddings](https://snap.stanford.edu/data/web-RedditEmbeddings.html) and put them into `/data`.

Then run preprocessing with:
```bash
python src/data/preprocess_graph.py
```

### Download preprocessed datasets

Since the scraping process has taken several days across several devices and clustering required a large pipeline, we have made the used datasets available on Google Drive. They are required in order to reproduce the `results.ipynb` notebook. All datasets are available on our google drive. They have all been used for different steps of the pipeline.
- [Dataset with clusters](https://drive.google.com/file/d/1zsww69--3KvZTzpohDfCLU15HBMIqDdj/view?usp=drive_link)
- [Enhanced Scraping Dataset](https://drive.google.com/file/d/12SAFinVcxSoniVVvr1bB94gKccXn8uBg/view?usp=drive_link)
- [Final Dataframe](https://drive.google.com/file/d/1qVLD5FblIiOlzK7Ux1mbyuu3fLP52Rd4/view?usp=drive_link)
 Before running the pipeline download both datasets and put them in `/data`. If you are interested in how the datasets have been obtained, check `/src/analysis/subreddit_clusters` (clustering script) and `/src/data/scraping` (scraping script).

## Project Structure

The directory structure of the project looks like this:

```
├── data/                       <- Raw and preprocessed data files
│
├── figures/                    <- Generated figures and visualizations
│
├── src/
│   ├── analysis/
│   │   ├── init_analysis/          <- Initial exploratory analysis
│   │   ├── init_analysis_enhanced/ <- Initial analysis with API enhanced data
│   │   ├── macro_analysis/         <- Macro-level analysis
│   │   ├── metric_analysis/        <- Metric analysis with Decision Trees and Random Forests
│   │   ├── sentiment_analysis/     <- Sentiment-specific analysis
│   │   ├── sentiment_graphs/       <- Sentiment graph creation and metrics
│   │   ├── subreddit_clustering/   <- Subreddit clustering
│   │   ├── temporal_analysis/      <- Temporal patterns analysis
│   │   ├── textual_analysis/       <- Analysis of title and body of posts
│   │   ├── top_subreddits/         <- Analysis of most popular subreddits
│   │   ├── upvote_analysis/        <- Upvote/engagement analysis
│   │   ├── propensity_analysis/    <- Propensity analysis of virality
│   │   ├── upvote_analysis/        <- Upvote/engagement analysis
│   │   ├── virality_analysis/      <- Analysis for virality
│   │   └── virality_graph/         <- Graphical analysis of virality
│   │
│   ├── data/                   <- Data and scraping utils
│   │
│   └── utils/                   <- Utility functions
│
└── results.ipynb               <- Main results notebook
```

## Contributions of Team Members
- Aziz Hamza: Temporal analysis and real life events linking
- Gal Gantar: Logistic regression for vitality prediction, Sentiment graphs, Centrality graphs, TF-IDF title analysis
- Jane Klavir: T-Testing and Data Story
- Jack Naimer: Clustering and Propensity Analysis of Virality
- Henrik Gruber: Data Scraping, Metric Creation and Analysis, Decision Trees, Random Forests and Minigames