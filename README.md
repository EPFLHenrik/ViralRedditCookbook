# Toxicstew: a cookbook on how to go viral on Reddit

TODO: add stuff we considered but dropped (eg all of the metrics in `create_sentiment_graph.py`)

## Abstract

TODO: Improve

*Virality on Reddit is often seen as luck, but patterns of language, timing, and community interaction shape what captures attention. ToxicStew: A Cookbook on How to Go Viral on Reddit will explore the underlying “recipe” for success by analyzing over one million crosslinking posts, enriched with engagement metrics such as upvotes, downvotes, and comment counts. The project will aim to uncover which combinations of factors—what subreddits to post in, which communities to crosslink, what tone or sentiment to adopt, and when to post—most strongly predict viral outcomes. By modeling these relationships and extracting interpretable patterns, it will reveal not only how virality emerges, but also how toxicity and controversy may serve as hidden ingredients in Reddit’s attention economy. Ultimately, the project will tell the story of how popularity can be engineered in online communities—and the social costs that may come with it.*

## Research questions

TODO: Improve
- Which factors most strongly predict virality on Reddit?
- How does sentiment and tone influence engagement?
- What role do subreddit interactions play in spreading content?
- When is the optimal time to post or crosslink for maximum visibility?
- Is virality associated with toxicity or polarization?


## Proposed additional datasets

TODO: Henrik & Gal
- Reddit embeddings dataset [link](https://snap.stanford.edu/data/web-RedditEmbeddings.html)


## Methods

TODO: Add
- Clustering
- Rolling averages
- Correlation coefficients
- Statistical tests ?
- Linear regression ?
- Logistic regression ?
- Matching ?

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

