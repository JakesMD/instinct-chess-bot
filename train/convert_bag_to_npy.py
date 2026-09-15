import chess, mmap, multiprocessing, os, time
import numpy as np
from apache_beam import coders
from instinct_chess_bot.utils import CONTEXT_SIZE, tokenize_sample

TRAIN_BAG_PATHS = [f"data/bags/train_action_value_{i:05d}.bag" for i in range(int(os.environ.get("NUM_TRAIN_BAGS")))]
VAL_BAG_PATHS = ["data/bags/test_action_value.bag"]

MAX_VAL_RECORDS = int(os.environ.get("MAX_VAL_RECORDS"))
MAX_TRAIN_RECORDS = int(os.environ.get("MAX_TRAIN_RECORDS"))
NUM_WORKERS = multiprocessing.cpu_count()


class BagRecords:
    '''
        Bag file format:
            [ record 0 ]                                    <- actual data
            [ ... ]
            [ record N-1 ]

            [ 8-byte pointer: where record 0 ends ]         <- index table
            [ ... ]
            [ 8-byte pointer: where record N-1 ends ]

            [ 8-byte pointer: where index table starts ]    <- last 8 bytes
    '''
    def __init__(self, filename: str):
        with open(filename, "rb") as file:
            # Lets us jump straight to any byte range on disk without reading the whole file.
            self._data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)

        # Record structure: FEN (string), move UCI (string), win probability (float from 0 to 1)
        self._decoder = coders.TupleCoder((coders.StrUtf8Coder(), coders.StrUtf8Coder(), coders.FloatCoder()))

        # Index table pointer is the last 8 bytes of the file.
        self._index_table_start = int.from_bytes(self._data[-8:], "little")

        # Each record is 8 bytes. Records end where index table starts.
        self._num_records = (self._data.size() - self._index_table_start) // 8

    def __len__(self) -> int:
        return self._num_records

    def __getitem__(self, record_index: int) -> tuple[str, str, float]:
        i = record_index.__index__()

        # Read the index table to find the start and end pointers.
        pointer_entry = self._index_table_start + i * 8
        record_start = 0 if i == 0 else int.from_bytes(self._data[pointer_entry - 8 : pointer_entry], "little")
        record_end = int.from_bytes(self._data[pointer_entry : pointer_entry + 8], "little")

        # Read and decode the record using the start and end pointers.
        raw = self._data[record_start:record_end]
        return self._decoder.decode(raw)


_job_counter = None
_total_jobs = None


def init_worker(job_counter, total_jobs):
    global _job_counter, _total_jobs
    _job_counter = job_counter
    _total_jobs = total_jobs


def convert_range(bag_path: str, start: int, end: int, x_path: str, y_path: str, offset: int):
    records = BagRecords(bag_path)
    x = np.lib.format.open_memmap(x_path, mode="r+")
    y = np.lib.format.open_memmap(y_path, mode="r+")

    start_time = time.time()
    for i in range(start, end):
        fen, action_uci, win_prob = records[i]
        board = chess.Board(fen)
        action = chess.Move.from_uci(action_uci)
        x[offset + i - start] = tokenize_sample(board, action).numpy()
        y[offset + i - start] = win_prob

    with _job_counter.get_lock():
        _job_counter.value += 1
        jobs_done = _job_counter.value

    print(f"Job {jobs_done}/{_total_jobs} done: {end - start} records ({(end - start) / (time.time() - start_time):.0f} records/sec)")


def convert(bag_paths: tuple[str], max_records: int, x_path: str, y_path: str) -> int:
    jobs = []
    total_records = 0
    remaining_records = max_records
    for bag_path in bag_paths:
        records_to_take = min(len(BagRecords(bag_path)), remaining_records)
        records_per_job = records_to_take // NUM_WORKERS + 1
        for start in range(0, records_to_take, records_per_job):
            jobs.append((bag_path, start, min(start + records_per_job, records_to_take), total_records + start))
        total_records += records_to_take
        remaining_records -= records_to_take
        if remaining_records <= 0:
            break

    np.lib.format.open_memmap(x_path, mode="w+", dtype=np.uint16, shape=(total_records, CONTEXT_SIZE))
    np.lib.format.open_memmap(y_path, mode="w+", dtype=np.float32, shape=(total_records,))

    print(f"{total_records} records, {len(jobs)} jobs, {NUM_WORKERS} workers")

    args = [(bag_path, start, end, x_path, y_path, offset) for bag_path, start, end, offset in jobs]
    job_counter = multiprocessing.Value("i", 0)
    with multiprocessing.Pool(NUM_WORKERS, initializer=init_worker, initargs=(job_counter, len(jobs))) as pool:
        pool.starmap(convert_range, args)

    return total_records


if __name__ == "__main__":
    start_time = time.time()

    total_val_records = convert(VAL_BAG_PATHS, MAX_VAL_RECORDS, "data/val_x.npy", "data/val_y.npy")
    print(f"Converted {total_val_records} val records")

    total_train_records = convert(TRAIN_BAG_PATHS, MAX_TRAIN_RECORDS, "data/train_x.npy", "data/train_y.npy")
    print(f"Converted {total_train_records} train records")

    print(f"Took {(time.time() - start_time) / 60:.1f} minutes")