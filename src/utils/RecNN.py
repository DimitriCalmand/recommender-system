import torch
import torch.nn as nn

class RecNN(nn.Module):
    def __init__(self):
        super(RecNN, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(8, 4),     # onehot_feat1
            nn.Embedding(30, 8),    # onehot_feat2
            nn.Embedding(1076, 32), # onehot_feat3
            nn.Embedding(12, 4),
            nn.Embedding(10, 4),
            nn.Embedding(3, 2),
            nn.Embedding(47, 6),
            nn.Embedding(340, 16),
            nn.Embedding(7, 4),
            nn.Embedding(5, 3),
            nn.Embedding(3, 2),
            nn.Embedding(2, 2),
            nn.Embedding(2, 2),
            nn.Embedding(2, 2),
            nn.Embedding(2, 2),
            nn.Embedding(2, 2),
            nn.Embedding(2, 2),
            nn.Embedding(2892, 32)  # video_tag_id, adjust range as needed
        ])


        total_emb_size = sum(emb.embedding_dim for emb in self.embeddings)

        self.model = nn.Sequential(
            nn.Linear(total_emb_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        for i, emb in enumerate(self.embeddings):
            max_index = emb.num_embeddings
            col_vals = x[:, i]
            if (col_vals >= max_index).any() or (col_vals < 0).any():
                print(f"[ERROR] Embedding {i}: min = {col_vals.min().item()}, max = {col_vals.max().item()}")

        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat = torch.cat(embedded, dim=1)
        return self.model(x_cat)