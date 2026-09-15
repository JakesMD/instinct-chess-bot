import copy, os, time, torch
import numpy as np
from instinct_chess_bot.model import InstinctTransformer
from instinct_chess_bot.utils import pick_device

MODEL_PATH = "weights/model.pt"
CHECKPOINT_PATH = "weights/model.ckpt.pt"

BATCH_SIZE = 1024
LEARNING_RATE = 3e-4
EVAL_EVERY_STEPS = 1000


def load_datasets(device: str):
    train_tokens = np.load("data/train_x.npy", mmap_mode="r")
    train_win_probs = np.load("data/train_y.npy", mmap_mode="r")
    val_tokens = np.load("data/val_x.npy", mmap_mode="r")
    val_win_probs = np.load("data/val_y.npy", mmap_mode="r")

    # Val data never changes, so tokenize and move it onto the device once up front.
    val_tokens = torch.from_numpy(val_tokens).long().to(device)
    val_win_probs = torch.from_numpy(val_win_probs).float().to(device)

    return train_tokens, train_win_probs, val_tokens, val_win_probs


def build_model(device: str):
    model = InstinctTransformer().to(device)
    train_model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=device == "cuda")
    return model, train_model, optimizer


def load_checkpoint(model: InstinctTransformer, optimizer: torch.optim.Optimizer, device: str):
    if not os.path.exists(CHECKPOINT_PATH):
        return 0, float("inf"), None

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step, best_val_loss, best_model_state = checkpoint["step"], checkpoint["best_val_loss"], checkpoint["best_model_state"]

    print(f"Resumed from {CHECKPOINT_PATH}: step {step}, best val loss {best_val_loss:.4f}")
    return step, best_val_loss, best_model_state


def save_checkpoint(model: InstinctTransformer, optimizer: torch.optim.Optimizer, step: int, best_val_loss: float, best_model_state: dict):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "best_model_state": best_model_state,
        },
        CHECKPOINT_PATH,
    )
    print(f"Checkpoint saved to {CHECKPOINT_PATH}.")


def train_step(train_tokens: np.ndarray, train_win_probs: np.ndarray, step: int, model, train_model, optimizer: torch.optim.Optimizer, device: str) -> torch.Tensor:
    batch_start = (step - 1) * BATCH_SIZE
    batch_end = batch_start + BATCH_SIZE
    batch_tokens = torch.from_numpy(train_tokens[batch_start:batch_end]).long().to(device)
    batch_win_probs = torch.from_numpy(train_win_probs[batch_start:batch_end]).float().to(device)

    model.train()
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        loss = model.loss(train_model(batch_tokens), batch_win_probs)
    loss.backward()
    optimizer.step()
    return loss


def run_eval(model, val_tokens, val_win_probs, device, best_val_loss: float, best_model_state):
    model.eval()
    val_count, total_loss = len(val_win_probs), 0.0
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        for batch_start in range(0, val_count, BATCH_SIZE):
            batch_tokens = val_tokens[batch_start : batch_start + BATCH_SIZE]
            batch_win_probs = val_win_probs[batch_start : batch_start + BATCH_SIZE]
            total_loss += model.loss(model(batch_tokens), batch_win_probs).item() * len(batch_win_probs)
    val_loss = total_loss / val_count

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())

    return val_loss, best_val_loss, best_model_state


def format_elapsed(start_time: float) -> str:
    seconds = time.time() - start_time
    hours, minutes = int(seconds // 3600), int(seconds % 3600 // 60)
    return f"{hours}h {minutes}m"


def log_progress(step: int, loss: torch.Tensor, val_loss: float, best_val_loss: float, last_step: int, last_time: float, start_time: float, current_time: float):
    steps_per_sec = (step - last_step) / (current_time - last_time)
    print(f"step {step} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | best val loss {best_val_loss:.4f} | {steps_per_sec:.2f} steps/s | elapsed {format_elapsed(start_time)}")


def save_final_model(best_model_state, best_val_loss: float, step: int):
    if best_model_state is not None:
        torch.save(best_model_state, MODEL_PATH)
        print(f"Saved best model to {MODEL_PATH} (best val loss {best_val_loss:.4f}, stopped after {step} steps).")


def train():
    start_time = time.time()
    device = pick_device()
    train_tokens, train_win_probs, val_tokens, val_win_probs = load_datasets(device)
    num_steps = len(train_win_probs) // BATCH_SIZE

    model, train_model, optimizer = build_model(device)
    step, best_val_loss, best_model_state = load_checkpoint(model, optimizer, device)

    last_step, last_time = step, time.time()

    try:
        while step < num_steps:
            step += 1
            loss = train_step(train_tokens, train_win_probs, step, model, train_model, optimizer, device)

            if step % EVAL_EVERY_STEPS == 0:
                current_time = time.time()
                val_loss, best_val_loss, best_model_state = run_eval(model, val_tokens, val_win_probs, device, best_val_loss, best_model_state)
                save_checkpoint(model, optimizer, step, best_val_loss, best_model_state)
                log_progress(step, loss, val_loss, best_val_loss, last_step, last_time, start_time, current_time)
                last_time = current_time
                last_step = step

    except KeyboardInterrupt:
        pass

    save_final_model(best_model_state, best_val_loss, step)

    print(f"Total training time: {format_elapsed(start_time)}")


if __name__ == "__main__":
    train()
