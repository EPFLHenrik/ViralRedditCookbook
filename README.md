# Toxicstew: a cookbook on how to go viral on Reddit

## Abstract

TODO: Improve

*Virality on Reddit is often seen as luck, but patterns of language, timing, and community interaction shape what captures attention. ToxicStew: A Cookbook on How to Go Viral on Reddit will explore the underlying “recipe” for success by analyzing over 800k crosslinking posts, enriched with engagement metrics such as upvotes, downvotes, and comment counts. The project will aim to uncover which combinations of factors (what subreddits to post in, which communities to crosslink, what tone or sentiment to adopt, and when to post) most strongly suggest viral outcomes. By modeling these relationships and extracting interpretable patterns, it will reveal not only how virality emerges, but also how toxicity and controversy may serve as hidden ingredients in Reddit’s attention economy. Ultimately, the project will tell the story of how popularity can be engineered in online communities.*

## Research questions

- **Primary**: Which combinations of linguistic, temporal, and network factors most suggest predict viral success (top 1% engagement) on Reddit?
- **Sentiment Analysis**: How do sentiment intensity, emotional toxicity, and aggressive communication patterns correlate with engagement metrics?
- **Network Effects**: What role do subreddit crosslinking patterns and community interactions play in content propagation?
- **Temporal Dynamics**: When is the optimal time to post or crosslink for maximum visibility and interactons across different subreddit communities?
- **Toxicity-Virality Relationship**: Is there a systematic relationship between controversial content and viral success, and what are the ethical implications?
- **Community-Specific Patterns**: How do viral patterns differ across subreddit communities with varying toxicity levels?


## Proposed additional datasets

TODO: Henrik & Gal
- Reddit embeddings dataset [link](https://snap.stanford.edu/data/web-RedditEmbeddings.html)

## Proposed Additional Datasets

### Primary Dataset: Reddit Hyperlinks Dataset
- **Source**: [Stanford SNAP Reddit Hyperlinks Dataset](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)
- **Size**: 860k hyperlinks between 51k subreddits
- **Format**: TSV files with source/target subreddits, timestamps, and post metadata
- **Processing**: NetworkX MultiDiGraph -> aggregated sentiment DiGraph

### Enrichment Dataset: Reddit Embeddings
- **Source**: [Stanford SNAP Reddit Embeddings](https://snap.stanford.edu/data/web-RedditEmbeddings.html)
- **Size**: 56k subreddit embeddings (300-dimensional vectors)
- **Format**: CSV with subreddit names and embedding vectors
- **Preprocesing**: Merge with primary dataset, perform clustering

## Reddit API Post Data Enrichment
- **Source**: Reddit API scraping of all 860k posts from hyperlinks dataset
- **Size**: Metadata of 860k posts obtained from Reddit API
- **Format**: Key metrics obtained 
    - `ups`: Upvotes (primary virality indicator)
    - `downs`: Downvotes (controversy measure)
    - `num_comments`: Comment count (engagement depth)
    - `score`: Net score (ups - downs)
    - `upvote_ratio`: Upvote percentage
- **Preprocessing**: Merging existing data with the primary dataset

## Data handling pipelines

TODO: improve

Raw data is uploaded to `/data` folder, the main preprocessing is done in `/src/data`, clustering of embeddings happens in `/src/analysis`.

## Methods

TODO: Add
- Clustering (Leiden clustering)
- Correlation coefficients
    - Pearson correlation
    - Sprearman correlation
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
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

