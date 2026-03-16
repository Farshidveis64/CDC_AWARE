# Datasets

The paper evaluates on six real-world datasets (Table 3).
Download each from its official source and place files as shown below.

---

## Expected folder structure

```
data/
├── twitter/
│   ├── graph.edgelist
│   └── node_attrs.json
├── stackoverflow/
│   ├── graph.edgelist
│   └── node_attrs.json
├── amazon/
│   ├── graph.edgelist
│   └── node_attrs.json
├── wikipedia/
│   ├── graph.edgelist
│   └── node_attrs.json
├── reddit/
│   ├── graph.edgelist
│   └── node_attrs.json
└── enron/
    ├── graph.edgelist
    └── node_attrs.json
```

---

## Edge list format

One edge per line:
```
<source_id>  <target_id>  <probability>
```
If probability is omitted, default `p = 0.1` is used.

---

## Node attributes format (`node_attrs.json`)

```json
{
  "0": {"verified": true,  "topic_similarity": 0.8, "karma": 0.6},
  "1": {"verified": false, "topic_similarity": 0.3, "karma": 0.2},
  ...
}
```

The attributes needed depend on the context type:

| Dataset       | Context type       | Required attributes                     |
|---------------|--------------------|-----------------------------------------|
| Twitter-Ads   | `verified_source`  | `verified` (bool)                       |
| StackOverflow | `tag_coherence`    | `tag_similarity` (float), `reputation`  |
| Amazon        | `category_path`    | `same_category` (bool), `review_quality`|
| Wikipedia     | `topic_coherence`  | `topic_similarity` (float)              |
| Reddit        | `subreddit_chain`  | `same_subreddit` (bool), `karma`        |
| Enron         | `thread_structure` | `in_thread` (bool), `thread_coherence`  |

---

## Download sources

| Dataset       | Nodes   | Edges     | Source |
|---------------|---------|-----------|--------|
| Twitter-Ads   | 81,306  | 1,768,149 | [SNAP ego-Twitter](http://snap.stanford.edu/data/ego-Twitter.html) |
| StackOverflow | 63,497  | 842,156   | [SNAP sx-stackoverflow](http://snap.stanford.edu/data/sx-stackoverflow.html) |
| Amazon        | 91,813  | 1,234,567 | [SNAP amazon0302](http://snap.stanford.edu/data/amazon0302.html) |
| Wikipedia     | 112,453 | 2,156,789 | [SNAP wiki-Talk](http://snap.stanford.edu/data/wiki-Talk.html) |
| Reddit        | 76,234  | 987,654   | [KONECT reddit_hyperlinks](http://konect.cc/networks/reddit_hyperlinks/) |
| Enron         | 36,692  | 367,662   | [SNAP email-Enron](http://snap.stanford.edu/data/email-Enron.html) |

---

## Quick download (Enron — smallest, good for testing)

```bash
mkdir -p data/enron
wget https://snap.stanford.edu/data/email-Enron.txt.gz
gunzip email-Enron.txt.gz
grep -v "^#" email-Enron.txt | awk '{print $1, $2, 0.1}' > data/enron/graph.edgelist
echo "Enron graph ready: $(wc -l < data/enron/graph.edgelist) edges"
```
