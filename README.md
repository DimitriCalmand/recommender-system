
Model Training:

2,5 methods:
- **Collaborative Filtering**: with als:
  - used metrics: interaction of user and watch ratio
  - results: 
    - `NDCG@10 = 0.84` - Best for position-aware ranking quality.
    - `MAP@10 = 0.76` – Great global ranking evaluation. 
    - `Precision@10 = 0.83` - Simple and intuitive.


- **Content-Based Filtering:** with Cosine Similarity
  - used metrics: Categories (encoded tags) in a similarity matrix
  - results:
    - `Precision@K = 0.06`
    - `Recall@K = 0.0014`
    - `HitRate@K = 0.45`

- **Hybridization:** with video popularity
  - Added weight in recommendations so it recommend more popular videos
  - results: 
    - `Precision@K = 0.12`
    - `Recall@K = 0.0031`
    - `HitRate@K = 0.69`

Also tried to add TF-IDF when doing the cosine similarity matrix but didn't improve the metrics

Why these approaches:
Training efficiency and inference speed
Match with the data we have 
Why two approaches:
Content-based filtering can do cold start.
Collaborative filtering can do high personalization
