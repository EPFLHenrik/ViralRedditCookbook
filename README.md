
# Longest path of shortest hate: Navigating the difficult landscapes of Reddit

## Abstract

TODO: Improve
*When using the Reddit we are often faced with the criticism from other users. In this project we set out to create a cheatsheet of rules you can apply to your behaviour on Reddit in order to either receive constructive criticism or just have fun browsing*

## Research questions

TODO: Improve
- Can posting the same type of post in two different subreddits receive substantially different reactions from other subreddits
- Are there subreddits who act as "hate path blockers" and tend to stay positive even when faced with negative crosslinks

## Proposed additional datasets

TODO: Henrik

## Methods

TODO: Add
- Clustering
- Rolling averages
- Statistical tests ?
- Linear regression ?
- Logistic regression ?
- Matching ?

## Proposed timeline

**October 2025**
- Proposals, literature review
- Exploratory data analysis (distributions)
- Data loading (NetworkX, Pandas)
- Graph data preprocessing (`MultiDiGraph` with all posts -> `DiGraph` with sentiments)
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
- [ ] Enriching data with torrent dataset (Henrik)
- [ ] Temporal analysis (Aziz)
- [ ] Clustering of embeddings (Jack)
- [ ] Top 10 subreddits analysis (Jane)
- [ ] Sentiment graph structure analysis (Gal)


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

