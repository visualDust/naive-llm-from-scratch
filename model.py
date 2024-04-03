import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class SelfAttentionHead(nn.Module):
    def __init__(
        self, head_size, block_size, n_embed, dropout_rate=0.2, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )  # no-look-ahead-mask
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head_size)
        B, T, C = x.shape
        # T is actually block size
        k = self.key(x)
        q = self.query(x)
        # conpute self attention scores:
        # dot product, and then scaling by 1/sqrt{length of a row in the keys or queries matrix}
        weighted_attention: torch.Tensor = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        # = (B,T,head_size)@(B,T,head_size).transpose(-2,-1)
        # = (B,T,head_size)@(B,head_size,T) -> (B,T,T)
        weighted_attention = weighted_attention.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # B,T,T, here T is block size
        weighted_attention = F.softmax(weighted_attention, dim=-1)
        weighted_attention = self.dropout(weighted_attention)
        # perform weighted aggregation of the values
        v = self.value(x)  # (B,T,head_size)
        out = weighted_attention @ v  # (B,T,T)@(B,T,head_size) -> (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self, n_head, head_size, block_size, n_embed, dropout_rate=0.2, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.heads = (
            nn.ModuleList(  # using ModuleList so that Heads are running in parallel
                [
                    SelfAttentionHead(
                        head_size=head_size, block_size=block_size, n_embed=n_embed
                    )
                    for _ in range(n_head)
                ]
            )
        )
        self.projection = nn.Linear(head_size * n_head, n_embed)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout_rate=0.2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, n_head, block_size, n_embed, **kwargs) -> None:
        super().__init__(**kwargs)
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(
            n_head=n_head, block_size=block_size, head_size=head_size, n_embed=n_embed
        )
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        y = self.sa(x)  # multi head attention
        x = self.ln1(x + y)  # residual (add and norm)
        y = self.ffwd(x)  # feedforward
        x = self.ln2(x + y)  # residual (add and norm)
        return x


class GPTLangModel(nn.Module):
    def __init__(
        self, vocabulary_size, n_decoder, n_embed, n_head, block_size, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[
                SelfAttentionBlock(
                    n_head=n_head, block_size=block_size, n_embed=n_embed
                )
                for _ in range(n_decoder)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocabulary_size)

    def _int_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, target=None):
        B, T = index.shape
        token_embedding = self.token_embedding_table(index)
        position_embedding = self.position_embedding_table(
            torch.arange(T, device=device)
        )
        x = token_embedding + position_embedding
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # if training, calculate loss
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = target.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -self.block_size :]
            # get the prediction
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # get the index(sample from the distribution)
            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1)  # (B, T+1)
        return index
