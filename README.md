# Toxicstew: a cookbook on how to go viral on Reddit

## Abstract

TODO: Improve

*Virality on Reddit is often seen as luck-based, but patterns of language, timing, and community interaction shape what captures attention. <ins>ToxicStew: A Cookbook on How to Go Viral on Reddit</ins> will explore the underlying “recipe” for success by analyzing over 850k crosslinking posts, enriched with engagement metrics such as upvotes, downvotes, and comment counts. The project will aim to uncover which combinations of factors (what subreddits to post in, which communities to crosslink, what tone or sentiment to adopt, and when to post) most strongly suggest viral outcomes. By modeling these relationships and extracting interpretable patterns, it will reveal not only how virality emerges, but also how toxicity and controversy may serve as hidden ingredients in Reddit’s attention economy. Additionally, through new language processing possibilities like LLMs, an analysis on hidden patterns to the human eye can be performed. Ultimately, the project will tell the story of how popularity can be engineered in online communities.*

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
- **Processing**: NetworkX MultiDiGraph -> aggregated sentiment DiGraph

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

#### Linear Regression
- **Purpose**: Fit a linear model to the data
- **Formula**: We find best $\beta_0, \ldots, \beta_N$ such that the error $\epsilon$ is the smallest possible with $Y = \beta_0 + \beta_1X_1 + ... + \beta_kX_k + \epsilon$
- **Implementation**: `smf.ols()`

#### Logistic Regression
- **Purpose**: Fit a linear regression to the data to predict the probability for the datapoint corresponding to class 1
- **Formula**: Find best $\beta_0, \ldots, \beta_N$ such that the $P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + ... + \beta_kX_k)}}$ gives the most accurate predictions
- **Implementation**: `smf.logit()`

TODO: Add
- Clustering (Leiden clustering)
- Correlation coefficients
    - Pearson correlation
    - Spearman correlation
- Hypothesis testing
    - t-test
    - KS test for normality (ktest_normal)
    - ...
- Linear regression
- Logistic regression
- Matching (Propensity score matching)

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


## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

