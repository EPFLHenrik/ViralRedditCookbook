TODO: perhaps move all figures to `/figures`

# Toxicstew: a cookbook on how to go viral on Reddit

## Abstract

TODO: Improve

*Virality on Reddit is often seen as luck-based, but patterns of language, timing, and community interaction shape what captures attention. <ins>ToxicStew: A Cookbook on How to Go Viral on Reddit</ins> will explore the underlying “recipe” for success by analyzing over 850k crosslinking posts, enriched with engagement metrics such as upvotes, downvotes, and comment counts. The project will aim to uncover which combinations of factors (what subreddits to post in, which communities to crosslink, what tone or sentiment to adopt, and when to post) most strongly suggest viral outcomes. By modeling these relationships and extracting interpretable patterns, it will reveal not only how virality emerges, but also how toxicity and controversy may serve as hidden ingredients in Reddit’s attention economy. Additionally, through new language processing possibilities like LLMs, an analysis on  patterns hidden to the human eye can be performed. Ultimately, the project will tell the story of how popularity can be engineered in online communities.*

## Research questions

- **Primary**: Which combinations of linguistic, temporal, and network factors most suggest predict viral success (top 1% engagement) on Reddit?
- **Sentiment Analysis**: How do sentiment intensity, emotional toxicity, writing styles and aggressive communication patterns correlate with engagement metrics?
- **Network Effects**: What role do subreddit crosslinking patterns and community interactions play in content propagation?
- **Temporal Dynamics**: When is the optimal time to post or crosslink for maximum visibility and interactons across different subreddit communities?
- **Toxicity-Virality Relationship**: Is there a systematic relationship between controversial content and viral success, and what are the ethical implications?
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

TODO: improve

Raw data is uploaded to `/data` folder, the main preprocessing is done in `/src/data`, clustering of embeddings happens in `/src/analysis`.

## Methods

TODO: improve
### Regression

#### Linear Regression
- **Purpose**: Fit a linear model to the data
- **Formula**: We find best $\beta_0, \ldots, \beta_N$ such that the error $\epsilon$ is the smallest possible with $Y = \beta_0 + \beta_1X_1 + ... + \beta_kX_k + \epsilon$
- **Implementation**: `smf.ols()`
- **Application in dataset**: Predict similarity between subreddits, number of posts overtime, post virality

#### Logistic Regression
- **Purpose**: Fit a linear regression to the data to predict the probability for the datapoint corresponding to class 1
- **Formula**: Find best $\beta_0, \ldots, \beta_N$ such that the $P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + ... + \beta_kX_k)}}$ gives the most accurate predictions
- **Implementation**: `smf.logit()`
- **Application in dataset**: Predict post sentiment

### Clustering

#### Leiden Clustering
- **Purpose**: Detect communities in a similarity graph (e.g., subreddit or user embeddings) by maximizing a quality function (typically modularity)
- **Formula**: Partition $G = (V, E)$ to maximize modularity $Q = \frac{1}{2m} \sum_{i,j} \bigg( A_{i,j} - \frac{k_ik_j}{2m} \bigg) \delta(c_i,c_j)$
- **Implementation**: `igraph.community_leiden()`
- **Application in dataset**: Identify clusters of subreddits with similar user bases and interaction patterns

### Correlation Coefficients

#### Pearson Correlation Coefficient
- **Purpose**: Measure the strength and direction of the linear relationship between two continuous variables
- **Formula**: $r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2} \sqrt{\sum_i (y_i - \bar{y})^2}}$ where $r \in [-1,1]$ with 1 = perfect positive linear correlation, -1 = perfect negative, 0 = no linear correlation
- **Implementation**: `scipy.stats.pearsonr()`
- **Application in dataset**: Calculating covariance between LIWC features among top subreddits, calculating pairwise covariance between subreddits' LIWC profiles

#### Spearman Correlation Coefficient
- **Purpose**: Assess the monotonic relationship between two ranked variables; captures nonlinear but monotonic associations
- **Formula**: $\rho = 1 - \frac{6\sum_i d_i^2}{n(n^2 - 1)}$ where $d_i$ is the rank difference for observation $i$, and $\rho \in [-1,1]$
- **Implementation**: `scipy.stats.spearmanr()`
- **Application in dataset**: Studying correlation between toxicity and virality, engagement and LIWC features, and temporal analysis

### Hypothesis Testing

#### t-Test
- **Purpose**: Test whether the means of two groups are significantly different
- **Formula**: (independent samples) $t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$ where $\bar{X}_1, \bar{X}_2$ are sample means, $s_1,s_2$ are standard deviations, and $n_1,n_2$ are sample sizes
- **Implementation**: `scipy.stats.ttest_ind()`
- **Application in dataset**: Determining whether posts are viral, whether controversy/negativity drives engagement

#### KS Test for Normality
- **Purpose**: Test whether a sample follows a specified continuous distribution (commonly the normal distribution)
- **Formula**: $D = \sup_x \, | F_n(x) - F(x) |$ where $F_n(x)$ is the empirical cumulative distribution function (ECDF) and $F(x)$ is the theoretical CDF
- **Implementation**: `scipy.stats.kstest()`
- **Application in dataset**: Comparing engagement / toxicity distributions between clusters

### Matching

#### Propensity Score Matching
- **Purpose**: Create balanced treatment and control groups by matching units with similar probabilities of receiving treatment, reducing selection bias in observational studies
- **Formula**: 
  1. Estimate propensity scores: $e(X_i) = P(T_i=1|X_i)$ using logistic regression
  2. Match treated units to control units with similar $e(X)$ using nearest neighbor
  3. Estimate treatment effect: $\hat{\tau}_{ATT} = \frac{1}{n_1}\sum_{i:T_i=1}[Y_i - Y_{j(i)}]$ where $j(i)$ is the matched control for treated unit $i$
- **Implementation**: `sklearn.linear_model.LogisticRegression(), sklearn.neighbors.NearestNeighbors()`
- **Application in dataset**: Comparing virality of toxic vs. non-toxic posts with similar baseline characteristics (LIWC features, subreddit, timing), isolate causal effect of toxicity on engagement by controlling for confounders (e.g. post length, readability, timestamp)


## Proposed timeline

**October 2025**
- Proposals, literature review
- Exploratory data analysis (distributions, missing values)
- Data loading (NetworkX, Pandas)
- Graph data preprocessing (`MultiDiGraph` with all posts -> `DiGraph` with aggregated sentiments)
- Dataset enrichment (TODO: Henrik)
- Clustering of nodes (TODO: Jack)
- Top 10 subreddits analysis (TODO: Jane)
- Temporal analysis (TODO: Aziz)
- Finising the `results.ipynb` notebook

**November 2025**
- In depth analysis of timing
- LLM analysis of enriched posts
- Hypothesis tests for all the proposed actions for more informative browsing
- Website
- Interactive Flourish visualizations

**December 2025**
- Polishing website
- Finishing up with all the experiments


## Internal milestones
- [x] Graph Data preprocessing (Gal)
- [x] Data loading (Gal, Henrik)
- [ ] Enriching data by scraping reddit (Henrik, Gal)
- [ ] Temporal analysis (Aziz)
- [ ] Clustering of embeddings (Jack)
- [ ] Top 10 subreddits analysis with hypothesis testing (Jane)
- [ ] Sentiment graph structure analysis (Gal)
- [ ] Analyze correlations in the data with personr and stuff we did at the tutorials (Somebody)
- [ ] Perform a bunch of T-tests, k_tests, ...

## How to use

Install packages

```bash
pip install -r pip_requirements.txt
```

If you don't use vscode you have to export `PYTHONPATH` environment variable to the root of the project for the relative imports to work
```bash
export PYTHONPATH=$(pwd)
```

Download the data

# TODO: our data is available on the Google drive

## Project Structure

The directory structure of the project looks like this:

```
├── data/                       <- Raw and preprocessed data files
│
├── figures/                    <- Generated figures and visualizations
│
├── src/
│   ├── analysis/
│   │   ├── init_analysis/      <- Initial exploratory analysis
│   │   ├── init_analysis_enhanced/ <- Initial analysis with API enhanced data
│   │   ├── macro_analysis/     <- Macro-level analysis
│   │   ├── sentiment_analysis/ <- Sentiment-specific analysis
│   │   ├── sentiment_graphs/  <- Sentiment graph creation and metrics
│   │   ├── subreddit_clustering/ <- Subreddit clustering
│   │   ├── temporal_analysis.py <- Temporal patterns analysis
│   │   ├── top_subreddits/     <- Analysis of most popular subreddits
│   │   └── upvote_analysis/    <- Upvote/engagement analysis
│   │
│   ├── data/                   <- Data and scrapping utils
│   │
│   └── utils/                   <- Utility functions
│
└── results.ipynb               <- Main results notebook
```

