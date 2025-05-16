# FinalProject_2025

## 📽️ Short Video Recommender System – KuaiRec

This project builds a personalized and scalable short video recommender system using the **KuaiRec dataset**, emulating the recommendation challenges faced by platforms like TikTok and Kuaishou. Our approach combines collaborative filtering, content-based filtering, hybridization, and neural models to address both personalization and cold-start issues.

---

## 🔍 Objective

Design and evaluate a system that predicts which short videos users are likely to engage with, based on their past interactions, preferences, and video metadata.

---

## 📦 Dataset

The **KuaiRec** dataset provides large-scale, fully observed user-video interactions from Kuaishou:

* User-item interactions (views, likes, durations, timestamps)
* Social network information
* Video metadata (tags, categories)
* Daily item features (e.g., popularity, play duration)

---

## 🧹 Data Processing

Performed in `data_processing.ipynb`:

* Handled missing values and unnecessary columns
* Parsed lists and converted timestamps to datetime
* Merged:

  * Social network data into `user_features`
  * Daily item features with `item_categories`

Saved outputs:

* `processed_user_features.csv`
* `processed_video_metadata.csv`
* `processed_interactions_train.csv`
* `processed_interactions_test.csv`

---

## 🛠 Feature Engineering

Performed in `feature_engineering.ipynb`. Extracted features to enrich modeling, support hybrid methods, and enable exploratory analysis.

### 👤 User Features

* `total_video_watched`
* `avg_watch_ratio`
* `preferred_category`

### 🎬 Video Features

* `total_likes`
* `avg_play_duration`
* `video_tags`

### 🔁 Interaction Features

* `user_video_watch_count`
* `user_video_avg_watch_ratio`

### 👥 Social Features

* `num_friends`
* `friends_favorite_categories`

Saved outputs:

* `user_features.csv`
* `video_metadata.csv`
* `interactions_train.csv`
* `interactions_test.csv`

---

## 🧠 Models & Methods

Trained and evaluated in `model_training.ipynb`.

### 1. **Collaborative Filtering (ALS)**

* Matrix factorization on user-item interaction matrix using `watch_ratio` as weight.
* **Why**: Learns latent patterns of user interests and content relevance.
* **Strengths**: High personalization and scalability.

### 2. **Content-Based Filtering**

* Cosine similarity between videos based on tag/category vectors.
* **Why**: Supports *cold start* users/items based on metadata.
* **Strengths**: No interaction history required.

### 3. **Hybrid Recommendation**

* Merges ALS predictions with video popularity scores.
* **Why**: Balances personalization with popularity trends for better engagement.
* **Strengths**: Improves diversity and relevance across user profiles.

### 4. **Neural Network**

* A feedforward model using one-hot encoded user features to predict `watch_ratio`.
* **Why**: Captures non-linear patterns and allows future extension (e.g., embeddings, sequences).
* **Note**: Current performance is limited, but results could be significantly improved by:

  * Using embedding layers
  * Incorporating interaction sequences (RNNs or Transformers)
  * Training on a larger subset of users

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