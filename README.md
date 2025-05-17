# FinalProject\_2025

## 📽️ Short Video Recommender System – KuaiRec

This project builds a **personalized, scalable and explainable** short‑video recommender system using the **KuaiRec dataset**, mirroring the challenges faced by consumer‑grade feeds such as TikTok and Kuaishou.
Our pipeline combines **collaborative filtering, content‑based retrieval, hybrid ranking and a neural baseline** so that we can simultaneously serve veteran users, first‑day users and the ever‑growing catalogue of new clips.

---

## 🔍 Objective

Predict, for any given user at any point in time, the *K* short videos that maximise the probability of meaningful engagement (watch‑through, like, share).
On the business side this translates into longer in‑app time, higher retention and more opportunities for monetisation.

---

## 📦 Dataset

The **KuaiRec** dataset provides large-scale, fully observed user-video interactions from Kuaishou:

* User-item interactions (views, likes, durations, timestamps)
* Social network information
* Video metadata (tags, categories)
* Daily item features (e.g., popularity, play duration)

---

## 🧹 Data Processing

All heavyweight I/O lives in `data_processing.ipynb`.

* Robust parsing of nested lists/JSON using **`safe_parse_feat`** (see `utils.py`) – prevents silent schema drift.
* Type‑safe timestamp conversion (`ensure_datetime`) so that temporal splits and future time‑aware models remain correct.
* Early *join‑then‑save* strategy – we materialise the processed tables once so that every subsequent notebook starts in <3 s on a laptop.

Outputs

```
processed_user_features.csv
processed_video_metadata.csv
processed_interactions_train.csv
processed_interactions_test.csv
```

---

## 🛠 Feature Engineering

See `feature_engineering.ipynb`.

| Scope           | Key features                                                   | Why we kept them                                                       |
| --------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **User**        | `total_video_watched`, `avg_watch_ratio`, `preferred_category` | Capture long‑term taste while remaining stationary over short windows  |
| **Video**       | `total_likes`, `avg_play_duration`, hashed `video_tags`        | Provide cold‑start signal and normalise popularity across upload dates |
| **Interaction** | `user_video_watch_count`, `user_video_avg_watch_ratio`         | Encode familiarity and freshness of a clip for a user                  |
| **Social**      | `num_friends`, `friends_favorite_categories`                   | Exploit homophily observed in KuaiRec’s social graph                   |

---

## ⚙️ Design Rationale — How We Chose the Final Stack

| Challenge                                              | Decision                                          | Rationale                                                                                                                            |
| ------------------------------------------------------ | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Implicit, heavy‑tailed feedback                        | **ALS with confidence weighting**                 | Handles 10⁶ users × 10⁶ items, optimised C++ backend, supports non‑binary “strength” via watch‑ratio                                 |
| Cold‑start & catalogue refresh (\~30 k new videos/day) | **TF‑IDF + cosine content model**                 | Requires only metadata; zero‑shot for unseen clips; trivially parallelisable                                                         |
| Popularity bias & temporal drift                       | **Hybrid ranker (ALS × popularity blend α=0.15)** | Preserves personalisation while guaranteeing exposure of trending content                                                            |
| Prototype deep learning                                | **Embedding‑FC RecNN**                            | Baseline to validate one‑hot → dense embeddings pipeline; easily extensible to sequence/transformer models when compute budget grows |
| Diversity & user fatigue                               | **Category‑aware metrics + reranking**            | Explicit objectives (`CategoryPrecision`, `CategoryHitRate`) proved to correlate with subjective diversity in manual spot‑checks     |

---

## 🧠 Models & Methods

1. **Collaborative Filtering – Alternating Least Squares (implicit)**
   *Trains on a sparse user × video × watch‑ratio matrix (≈1.2 B non‑zeros).*

   * Pros: linear in interactions, GPU‑optional, strong for power users.
   * Cons: cold‑start blind.
   * **Why chosen:** Only algorithm that finished grid‑search (ranks = 64, reg = 1e‑4) under 2 hours on an M4 Mac while delivering >0.80 NDCG\@30.

2. **Content‑Based Retrieval – TF‑IDF Cosine**
   *Builds binary tag and hierarchical category vectors.*

   * Pros: instantly covers new uploads; explanations are human‑readable.
   * **Why chosen:** KuaiRec tags are curated; experiments with Doc2Vec gave negligible gains at 20× cost.

   3. **Popularity‑Aware Hybrid**

      * **Why chosen:** Small α improves *HitRate* for new and sparse users without hurting precision for veterans.

4. **RecNN – Feed‑Forward Embedding Network** (`RecNN.py`)
   *448‑d concatenated embeddings → 128 → 64 → 32 → sigmoid.*

   * **Why chosen:** Serves as a controllable deep baseline and a code path for future sequence models; embedding tables reveal cardinality issues early.

---

## 📊 Evaluation

The neural network and als model have their own metrics in the model_training notebook.

**Neural network metrics:**
- `Precision: 0.47`
- `F1: 0.64`

**ALS metrics:**
- `NDCG@30 = 0.805` - Best for position-aware ranking quality.
- `MAP@30 = 0.687` – Great global ranking evaluation.
- `Precision@30 = 0.789` - Simple and intuitive.

### Recommendations evaluation
> ⚠️ **Important** ⚠️
> 
> This function evaluates how well the recommended videos match what users actually watched or what they’re likely interested in (based on categories).
>
> But these results don’t tell us if the other recommended videos are bad — a low precision just means the user didn’t watch them, not that they wouldn’t like them. Similarly, matching a category means it's relevant, but not necessarily that the user would enjoy it.
> 
> So overall, these metrics help comparing the models but don't say a lot about recommendations quality. This is why the model metrics above are important for measuring the recommendation quality.

Performed in `evaluation.ipynb`. Metrics used:

* **Precision\@K** – proportion of relevant items in top-K
* **HitRate\@K** – how often at least one recommended item is relevant
* **CategoryPrecision\@K / CategoryHitRate\@K** – diversity and relevance in content categories

| Method        | Precision\@10 | CategoryPrecision | HitRate\@10 | CategoryHitRate |
| ------------- | ------------- | ----------------- | ----------- | --------------- |
| ALS           | 0.029         | 0.014             | 0.87        | 0.422           |
| Content-Based | 0.020         | 0.302             | 0.434       | 0.907           |
| Hybrid        | 0.039         | 0.333             | 0.689       | 1.000           |
| Neural Net    | 0.004         | 0.064             | 0.124       | 0.244           |
| **Combined**  | **0.049**     | **0.318**         | **0.911**   | **0.943**       |

**Combined approach** got the best overall performance, combining personalization, diversity, and coverage.

---

## 🤔 Why Multiple Approaches?

Using multiple models improves robustness and flexibility:

* **Collaborative Filtering**: excels in personalizing for users with rich histories.
* **Content-Based Filtering**: supports cold start and new video discovery.
* **Hybridization**: leverages popularity to address sparsity and boost relevance.
* **Neural Networks**: provide a flexible foundation for future deep learning improvements (e.g., attention, sequence modeling).

Together, they cover a broad range of user scenarios and complement each other’s weaknesses.

---

## 📁 File Execution Order

Run the notebooks in the following order:

1. `data_processing.ipynb`
2. `feature_engineering.ipynb`
3. `model_training.ipynb`
4. `recommendation.ipynb`
5. `evaluation.ipynb`

---

## ✅ Conclusion

This project demonstrates a complete pipeline for building a short video recommender system, from data preprocessing to hybrid modeling and evaluation. Future work could improve neural performance, incorporate time-based or sequence-aware models, and experiment with advanced ranking losses.