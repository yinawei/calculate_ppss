import json
import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import gc
from scipy.spatial import KDTree, cKDTree
import pickle
import time
import shutil
from collections import OrderedDict, defaultdict
from pathlib import Path


# ==================== Global variables for worker processes ====================
GLOBAL_METADATA = None
GLOBAL_TILE_CACHE = None
GLOBAL_DENDRITE_CACHE = None

# Float64 coordinates preserve the distance values computed by the original
# implementation.  Dendrite codes make the binary tile files compact.
TILE_DTYPE = np.dtype(
    [("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
     ("dendrite_code", "<i4"), ("den_id", "<i8")]
)


def readSWC(swc_path, mode='simple'):
    n_skip = 0
    with open(swc_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("#"):
                n_skip += 1
            else:
                break

    names = ["##n", "type", "x", "y", "z", "r", "parent"]
    used_cols = [0, 1, 2, 3, 4, 5, 6]
    if mode == 'simple':
        pass
    df = pd.read_csv(swc_path, index_col=0, skiprows=n_skip, sep=" ",
                     usecols=used_cols,
                     names=names
                     )
    return df


# ==================== Bouton Read Function ====================
def read_axon(swc_path, mode='simple'):
    n_skip = 0
    with open(swc_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("#"):
                n_skip += 1
            else:
                break
    names = ["x", "y", "z"]
    used_cols = [0, 1, 2]
    if mode == 'simple':
        pass
    df = pd.read_csv(swc_path, skiprows=n_skip+1, sep=" ",
                     usecols=used_cols,
                     names=names
                     )
    return df


def _tile_key(coords, tile_size):
    return np.floor(coords / tile_size).astype(np.int64)


class _LRUFileHandles:
    """Bound the number of tile files open while constructing the index."""

    def __init__(self, limit=128):
        self.limit = limit
        self.handles = OrderedDict()

    def get(self, path):
        path = str(path)
        handle = self.handles.pop(path, None)
        if handle is None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            handle = open(path, "ab")
        self.handles[path] = handle
        if len(self.handles) > self.limit:
            _, old_handle = self.handles.popitem(last=False)
            old_handle.close()
        return handle

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def _index_metadata_path(index_dir):
    return Path(index_dir) / "metadata.json"


def build_spatial_index(dendrite_dir, index_dir, tile_size=100.0,
                        rebuild=False):
    """Build reusable binary spatial tiles from all dendrite SWCs.

    The build streams one dendrite at a time and therefore does not require all
    dendrite points to fit in memory.
    """
    dendrite_dir = Path(dendrite_dir).resolve()
    index_dir = Path(index_dir).resolve()
    metadata_path = _index_metadata_path(index_dir)
    swc_paths = sorted(dendrite_dir.glob("*.swc"))
    dendrite_ids = [path.stem for path in swc_paths]

    if metadata_path.exists() and not rebuild:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        compatible = (
            metadata.get("complete") is True
            and metadata.get("dendrite_dir") == str(dendrite_dir)
            and metadata.get("tile_size") == float(tile_size)
            and metadata.get("swc_count") == len(swc_paths)
            and metadata.get("record_size") == TILE_DTYPE.itemsize
            and metadata.get("dendrite_ids") == dendrite_ids
        )
        if compatible:
            return metadata
        raise RuntimeError(
            "Existing spatial index does not match the requested data. "
            "Set rebuild_index = True in the main program and run again."
        )

    if rebuild and index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = index_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "complete": False,
        "dendrite_dir": str(dendrite_dir),
        "tile_size": float(tile_size),
        "swc_count": len(swc_paths),
        "record_size": TILE_DTYPE.itemsize,
        "dendrite_ids": dendrite_ids,
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)

    handles = _LRUFileHandles()
    point_count = 0
    try:
        for dendrite_code, swc_path in enumerate(
                tqdm(swc_paths, desc="Building spatial index")):
            swc = readSWC(swc_path)
            den = swc.loc[swc.type.isin([3, 4]), ["x", "y", "z"]]
            if den.empty:
                continue
            coords = den.to_numpy(dtype=np.float64, copy=False)
            keys = _tile_key(coords, tile_size)

            # Group before appending so each tile needs only one write per SWC.
            unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
            node_ids = den.index.to_numpy(dtype=np.int64, copy=False)
            for group_no, key in enumerate(unique_keys):
                selected = np.flatnonzero(inverse == group_no)
                records = np.empty(len(selected), dtype=TILE_DTYPE)
                records["x"] = coords[selected, 0]
                records["y"] = coords[selected, 1]
                records["z"] = coords[selected, 2]
                records["dendrite_code"] = dendrite_code
                records["den_id"] = node_ids[selected]
                tile_path = tiles_dir / f"{key[0]}_{key[1]}_{key[2]}.bin"
                records.tofile(handles.get(tile_path))
                point_count += len(records)
    finally:
        handles.close()

    metadata["complete"] = True
    metadata["point_count"] = point_count
    metadata["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


class TileCache:
    """Per-worker LRU cache containing local tile arrays and cKDTrees."""

    def __init__(self, tiles_dir, max_tiles=64):
        self.tiles_dir = Path(tiles_dir)
        self.max_tiles = max_tiles
        self.cache = OrderedDict()

    def get(self, key):
        key = tuple(int(value) for value in key)
        cached = self.cache.pop(key, None)
        if cached is not None:
            self.cache[key] = cached
            return cached

        path = self.tiles_dir / f"{key[0]}_{key[1]}_{key[2]}.bin"
        if not path.exists():
            return None
        records = np.fromfile(path, dtype=TILE_DTYPE)
        coords = np.column_stack(
            (records["x"], records["y"], records["z"])
        )
        cached = (records, coords, cKDTree(coords))
        self.cache[key] = cached
        if len(self.cache) > self.max_tiles:
            self.cache.popitem(last=False)
        return cached


class DendriteTreeCache:
    """Small exact-KDTree cache used only for spatially discovered matches.

    Re-querying the matched dendrite with the same scipy KDTree implementation
    as the reference script reproduces its choice when duplicate/equidistant
    dendrite nodes exist.  Unmatched dendrites are never loaded here.
    """

    def __init__(self, dendrite_dir, dendrite_ids, max_trees=64):
        self.dendrite_dir = Path(dendrite_dir)
        self.dendrite_ids = dendrite_ids
        self.max_trees = max_trees
        self.cache = OrderedDict()

    def get(self, dendrite_code):
        dendrite_code = int(dendrite_code)
        cached = self.cache.pop(dendrite_code, None)
        if cached is not None:
            self.cache[dendrite_code] = cached
            return cached
        dendrite_id = self.dendrite_ids[dendrite_code]
        swc = readSWC(self.dendrite_dir / f"{dendrite_id}.swc")
        den = swc.loc[swc.type.isin([3, 4]), ["x", "y", "z"]].astype(float)
        cached = (den, KDTree(den.to_numpy(dtype=np.float64, copy=False)))
        self.cache[dendrite_code] = cached
        if len(self.cache) > self.max_trees:
            self.cache.popitem(last=False)
        return cached


def init_worker(index_dir, max_cached_tiles):
    global GLOBAL_METADATA, GLOBAL_TILE_CACHE, GLOBAL_DENDRITE_CACHE
    with open(_index_metadata_path(index_dir), "r", encoding="utf-8") as handle:
        GLOBAL_METADATA = json.load(handle)
    GLOBAL_TILE_CACHE = TileCache(
        Path(index_dir) / "tiles", max_tiles=max_cached_tiles
    )
    GLOBAL_DENDRITE_CACHE = DendriteTreeCache(
        GLOBAL_METADATA["dendrite_dir"],
        GLOBAL_METADATA["dendrite_ids"],
    )


def _relevant_tiles(axon_coords, radius, tile_size):
    """Map each intersected tile to the axon-row positions querying it."""
    tile_to_rows = defaultdict(list)
    EPS = 1e-10
    lower = np.floor((axon_coords - radius - EPS) / tile_size).astype(np.int64)
    upper = np.ceil((axon_coords + radius + EPS) / tile_size).astype(np.int64)
    for row_no, (lo, hi) in enumerate(zip(lower, upper)):
        for tx in range(lo[0], hi[0] + 1):
            for ty in range(lo[1], hi[1] + 1):
                for tz in range(lo[2], hi[2] + 1):
                    tile_to_rows[(tx, ty, tz)].append(row_no)
    return tile_to_rows


def func_worker(axon_path, dest, distance_threshold=5.0):
    """Calculate PPSS for one axon using nearby spatial tiles only."""
    axon_path = Path(axon_path)
    axon_filename = axon_path.name
    current_dendrite_id = None
    try:
        axon_df = read_axon(axon_path)
        axon = axon_df[["x", "y", "z"]].astype(float)
        if axon.empty:
            return axon_filename, True, "Processed spatial tiles, matches 0"

        coords = axon.to_numpy(dtype=np.float64, copy=False)
        tile_size = float(GLOBAL_METADATA["tile_size"])
        dendrite_ids = GLOBAL_METADATA["dendrite_ids"]
        # The spatial stage discovers candidates only.  A tiny margin prevents
        # floating-point boundary effects from discarding a true match before
        # the exact scipy KDTree query below.  Final acceptance still uses the
        # original distance_threshold exactly.
        candidate_radius = (
            distance_threshold
            + max(1e-9, abs(distance_threshold) * 1e-12)
        )
        tile_to_rows = _relevant_tiles(coords, candidate_radius, tile_size)

        # key=(axon row position, dendrite code), value=(distance, den_id).
        # Strict '<' matches the single-nearest-neighbour behaviour.  The den-id
        # tie-break makes results deterministic for exactly equidistant points.
        best = {}
        for tile_key in sorted(tile_to_rows):
            cached = GLOBAL_TILE_CACHE.get(tile_key)
            if cached is None:
                continue
            records, tile_coords, tree = cached
            row_positions = np.asarray(tile_to_rows[tile_key], dtype=np.int64)
            neighbours = tree.query_ball_point(
                coords[row_positions], r=candidate_radius, workers=1
            )
            for local_row, point_indices in enumerate(neighbours):
                axon_row = int(row_positions[local_row])
                if not point_indices:
                    continue
                point_indices = np.asarray(point_indices, dtype=np.int64)
                deltas = tile_coords[point_indices] - coords[axon_row]
                distances = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
                for point_index, distance in zip(point_indices, distances):
                    dendrite_code = int(records["dendrite_code"][point_index])
                    den_id = int(records["den_id"][point_index])
                    key = (axon_row, dendrite_code)
                    old = best.get(key)
                    candidate = (float(distance), den_id)
                    if old is None or candidate < old:
                        best[key] = candidate

        by_dendrite = defaultdict(list)
        axon_node_ids = axon.index.to_numpy()
        for (axon_row, dendrite_code), (distance, den_id) in best.items():
            by_dendrite[dendrite_code].append(
                (axon_row, axon_node_ids[axon_row], den_id, distance)
            )

        Path(dest).mkdir(parents=True, exist_ok=True)
        for dendrite_code, rows in by_dendrite.items():
            rows.sort(key=lambda row: row[0])
            dendrite_id = dendrite_ids[dendrite_code]
            current_dendrite_id = dendrite_id
            row_positions = np.asarray([row[0] for row in rows], dtype=np.int64)

            # Exact refinement with the same per-dendrite KDTree query as the
            # reference implementation.  The spatial stage has already proved
            # that these pairs are within the radius, so only true matches pay
            # this cost.
            den, exact_tree = GLOBAL_DENDRITE_CACHE.get(dendrite_code)
            exact_distances, exact_indices = exact_tree.query(
                coords[row_positions], p=2,
                distance_upper_bound=distance_threshold,
            )
            exact_distances = np.asarray(exact_distances, dtype=np.float64)
            exact_indices = np.asarray(exact_indices, dtype=np.int64)

            # A query with no exact neighbour returns distance == inf and
            # index == len(den).  Filter it exactly as the original code does
            # instead of indexing den.iloc out of bounds and aborting the
            # remaining dendrites for this bouton file.
            valid = (
                np.isfinite(exact_distances)
                & (exact_distances <= distance_threshold)
                & (exact_indices >= 0)
                & (exact_indices < len(den))
            )
            if not np.any(valid):
                continue

            valid_rows = [
                row for row, keep in zip(rows, valid) if keep
            ]
            exact_distances = exact_distances[valid]
            exact_indices = exact_indices[valid]
            exact_den_ids = den.iloc[exact_indices].index.to_numpy()
            output_path = Path(dest) / (
                f"{axon_filename}_{dendrite_id}.csv"
            )
            pd.DataFrame(
                {
                    "axon_id": [row[1] for row in valid_rows],
                    "den_id": exact_den_ids,
                    "dis": exact_distances,
                }
            ).to_csv(output_path)
        gc.collect()
        return axon_filename, True, f"Processed spatial tiles, matches {len(by_dendrite)}"
    except Exception as exc:
        return (
            axon_filename,
            False,
            f"dendrite={current_dendrite_id}, "
            f"{type(exc).__name__}: {exc}"
        )


# ==================== Custom Progress Callback ====================
class ProgressTracker:
    """Track progress across multiple processes"""
    def __init__(self, total, desc="Processing"):
        self.total = total
        self.desc = desc
        self.pbar = None
        self.completed = 0
        self.start_time = time.time()

    def update(self, result):
        """Update progress bar with result"""
        if self.pbar is None:
            self.pbar = tqdm(total=self.total, desc=self.desc)

        self.completed += 1
        self.pbar.update(1)

        # Display additional info if available
        if isinstance(result, tuple) and len(result) >= 3:
            axon_id, success, message = result
            if success and "matches" in message:
                match_info = message.split('matches ')[-1].split()[0]
                self.pbar.set_postfix({
                    'current': axon_id[:30],
                    'matches': match_info
                })

        # Close when done
        if self.completed >= self.total:
            self.pbar.close()
            elapsed = time.time() - self.start_time
            print(f"\nCompleted in {elapsed/60:.2f} minutes")


# ==================== Main Program ====================
if __name__ == "__main__":
    total_start_time = time.time()

    FULL = True

    if FULL:
        # use complete neuron axon morphology data with bouton coordinates
        axon_src = ['./data/predicted_bouton_locations/bouton_type5']

        # use complete neuron dendrite morphology data
        src2 = './data/155k_den_1um/1um/SEU-ALLEN_local_SWC_CCFv3'
        dest = './results/results_pb_optimized_full'
        log_path = './log/log_pb_optimized_full'
        index_path = './cache/spatial_dendrite_tiles_full'
    else:
        # use example neuron axon morphology data with bouton coordinates
        axon_src = ['./data/axon_bouton_example']

        # use example neuron dendrite morphology data
        src2 = './data/dendrite_example'
        dest = './results/results_pb_optimized'
        log_path = './log/log_pb_optimized'
        index_path = './cache/spatial_dendrite_tiles'
    
    # Get file lists
    swc_path_list1 = []
    for src in axon_src:
        swc_list1 = os.listdir(src)
        swc_path_list1.extend([src + '/' + i for i in swc_list1])
    swc_path_list1 = sorted(swc_path_list1)
    
    swc_list2 = os.listdir(src2)
    swc_path_list2 = [i[:-4] for i in swc_list2 if i.endswith('.swc')]
    swc_path_list2 = sorted(swc_path_list2)
    
    distance_threshold = 5.0
    tile_size = 100.0
    max_cached_tiles = 64
    rebuild_index = True
    
    cores = min(8, multiprocessing.cpu_count())  # int(multiprocessing.cpu_count() * 0.8)
    print("FULL: ", FULL)
    print("cores: ", cores)
    
    os.makedirs(dest, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(index_path, exist_ok=True)
    
    # ==================== Load/Build Spatial Dendrite Index ====================
    # Optimization: replace one KDTree per dendrite with spatial tiles.  Only
    # this cache block and func_worker differ algorithmically from the original.
    print("=" * 50)
    print("Loading/Building spatial dendrite index...")
    print("=" * 50)
    index_start_time = time.time()
    metadata = build_spatial_index(
        src2, index_path, tile_size=tile_size, rebuild=rebuild_index
    )
    print(f"Loaded {metadata['swc_count']} dendrites, "
          f"{metadata.get('point_count', 'cached')} dendrite points")
    index_elapsed = time.time() - index_start_time
    print(f"Spatial index completed in {index_elapsed/60:.2f} minutes")
    
    # ==================== Multi-processing with Progress ====================
    print("=" * 50)
    print("Processing (1 axon vs nearby dendrites)...")
    print("Each worker reads its own axon SWC file")
    print("=" * 50)
    
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    
    # Filter already processed axons
    pending_axons = []
    for path_swc_i in swc_path_list1:
        swc_i = path_swc_i.split('/')[-1]
        log_file = os.path.join(log_path, swc_i)
        if not os.path.exists(log_file):
            pending_axons.append(path_swc_i)
    
    print(f"Processing {len(pending_axons)} axons (skipping {len(swc_path_list1) - len(pending_axons)} already done)")
    
    if pending_axons:
        # Create progress tracker
        progress = ProgressTracker(total=len(pending_axons), desc="Processing axons")
        
        # Create process pool
        pool = multiprocessing.Pool(
            processes=cores,
            initializer=init_worker,
            initargs=(str(Path(index_path).resolve()), max_cached_tiles)
        )
        
        # Submit tasks with async
        tasks = []
        for axon_path in pending_axons:
            tasks.append(pool.apply_async(
                func_worker, (axon_path, dest, distance_threshold)
            ))
        
        # Collect results with progress
        for task in tasks:
            try:
                result = task.get(timeout=3600)  # 1 hour timeout
                progress.update(result)
                
                # Mark as completed
                if result and result[1]:  # success
                    axon_id = result[0]
                    os.makedirs(os.path.join(log_path, axon_id), exist_ok=True)
                else:
                    print(f"\nFailed: {result}")
            except multiprocessing.TimeoutError:
                print("\nTask timeout!")
            except Exception as e:
                print(f"\nTask failed: {e}")
        
        pool.close()
        pool.join()
        
        # Close progress bar
        if progress.pbar:
            progress.pbar.close()
    
    total_elapsed = time.time() - total_start_time
    print("=" * 50)
    print("Processing complete!")
    print(f"Total elapsed time: {total_elapsed/60:.2f} minutes")
    print("=" * 50)
