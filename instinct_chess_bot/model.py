import chess, torch
import torch.nn as nn
import torch.nn.functional as F
from instinct_chess_bot.utils import NUM_TOKENS, CONTEXT_SIZE, tokenize_sample

NUM_BINS = 128
NUM_HEADS = 8
NUM_LAYERS = 8
EMBEDDING_DIM = 256
FEEDFORWARD_DIM = 4 * EMBEDDING_DIM

class InstinctTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(NUM_TOKENS, EMBEDDING_DIM)
        self.positional_embedding = nn.Parameter(torch.randn(1, CONTEXT_SIZE, EMBEDDING_DIM) * 0.02)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=EMBEDDING_DIM,
                nhead=NUM_HEADS,
                dim_feedforward=FEEDFORWARD_DIM,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=NUM_LAYERS
        )

        self.output_head = nn.Linear(EMBEDDING_DIM, NUM_BINS)

        # Store the bin centers on the GPU/CPU.
        bin_edges = torch.linspace(0.0, 1.0, NUM_BINS + 1)
        self.register_buffer("bin_centers", (bin_edges[:-1] + bin_edges[1:]) / 2, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(tokens) + self.positional_embedding
        hidden = self.encoder(hidden)

        # Only feed the last token's vector (the move) to the output head. It now encodes the whole board.
        return self.output_head(hidden[:, -1, :])

    def loss(self, logits: torch.Tensor, win_prob: torch.Tensor) -> torch.Tensor:
        bin_index = (win_prob * NUM_BINS).long().clamp(max=NUM_BINS - 1)
        return F.cross_entropy(logits, bin_index)

    def expected_action_value(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.bin_centers).sum(dim=-1)

    def pick_move(self, board: chess.Board, device: str = "cpu") -> chess.Move:
        legal_moves = list(board.legal_moves)
        batch = torch.stack([tokenize_sample(board, move) for move in legal_moves]).to(device)
        with torch.no_grad():
            win_probs = self.expected_action_value(self(batch))
        best_move_index = int(torch.argmax(win_probs).item())
        return legal_moves[best_move_index]
