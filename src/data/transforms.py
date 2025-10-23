import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def static_csv_to_tensor_vectorized(path: Path, rows_per_frame) -> np.ndarray:
    df = pd.read_csv(path, header=None, usecols=[0, 1, 2, 3, 4, 5],
                    names=['side', 'sensor', 'b0', 'b1', 'b2', 'b3'])
    total_frames = len(df) // rows_per_frame
    df = df.iloc[:total_frames * rows_per_frame]

    frame_indices = df.index.to_numpy() // rows_per_frame
    sensor_indices = df['sensor'].to_numpy()
    height_base_indices = df['side'].map({'left': 0, 'right': 4}).to_numpy()
    bits = df[['b0', 'b1', 'b2', 'b3']].to_numpy(dtype=np.uint8)

    x = np.zeros(shape=(total_frames, 1, 8, 30), dtype=np.uint8)
    height_indices = height_base_indices[:, np.newaxis] + np.arange(4)
    x[frame_indices[:, np.newaxis], 0, height_indices, sensor_indices[:, np.newaxis]] = bits

    return x

def preprocess_static_data(data_roots, preprocessed_data_roots, rows_per_frame):
    print("--- Starting Data Pre-processing (Vectorized) ---")
    for i in range(len(data_roots)):
        raw_path = Path(data_roots[i])
        processed_path = Path(preprocessed_data_roots[i])
        processed_path.mkdir(exist_ok=True)

        csv_files = list(raw_path.glob('*.csv'))
        for csv_file in tqdm(csv_files, desc="Converting CSV to .npy"):
            npy_filename = processed_path / f"{csv_file.stem}.npy"
            if not npy_filename.exists():
                full_tensor = static_csv_to_tensor_vectorized(path=csv_file, rows_per_frame=rows_per_frame)
                np.save(npy_filename, full_tensor)

    print("--- Pre-processing Complete ---")
    return None

"""

def dynamic_csv_to_tensor_vectorized(path: Path, rows_per_frame):
    df = pd.read_csv(path)

    # --------- 1) 프레임 단위 시간 / 좌표 ---------
    # 60행마다 한 프레임
    total_frames = len(df) // rows_per_frame
    df = df.iloc[:total_frames * rows_per_frame]

    # 프레임 인덱스
    frame_idx = df.index.to_numpy() // rows_per_frame

    # 각 프레임의 시간(elapsed_s) 추출 (첫 행만 사용)
    times = df.groupby(frame_idx)["elapsed_s"].last().to_numpy()

    # 각 프레임의 좌표(x,y,z) 추출 (첫 행만 사용)
    coords = df.groupby(frame_idx)[["x", "y", "z"]].last().to_numpy(dtype=np.float32)

    # --------- 2) 좌표 보간 (piecewise linear) ---------
    # x,y,z가 변화하는 지점을 잡고 그 사이 구간은 시간 기준으로 선형 보간
    # 변화 검출
    change_mask = np.any(np.diff(coords, axis=0) != 0, axis=1)
    change_points = np.concatenate([[0], np.where(change_mask)[0] + 1, [total_frames-1]])
    change_points = np.unique(change_points)

    # 보간 수행
    coords_interp = np.zeros_like(coords)
    for i in range(len(change_points)-1):
        s, e = change_points[i], change_points[i+1]
        t0, t1 = times[s], times[e]
        p0, p1 = coords[s], coords[e]
        if t1 == t0:  # 시간 같으면 그대로
            coords_interp[s:e+1] = p0
        else:
            alpha = (times[s:e+1] - t0)[:,None] / (t1 - t0)
            coords_interp[s:e+1] = (1-alpha)*p0 + alpha*p1

    min_vals = coords_interp.min(axis=0)   # (3,)
    max_vals = coords_interp.max(axis=0)   # (3,)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # 분모 0 방지

    coords_scaled = (coords_interp - min_vals) / range_vals

    # --------- 3) 센서 비트 -> (8,30) 이미지 ---------
    # side/b0..b3 를 이미지로 매핑
    side = df["left/right"].map({"left":0,"right":1}).to_numpy()
    sensor = df["count"].to_numpy() % 30   # 0~29
    bits = df[["b0","b1","b2","b3"]].to_numpy(dtype=np.uint8)

    height_base = side * 4
    height_idx = height_base[:,None] + np.arange(4)
    frame_idx = frame_idx[:,None]

    X = np.zeros((total_frames, 8, 30), dtype=np.uint8)
    X[frame_idx, height_idx, sensor[:,None]] = bits
    X = X[:, np.newaxis, :, :]   # 또는 X = X.reshape(total_frames, 1, 8, 30)


    return X, coords_scaled


def preprocess_dynamic_data(data_roots, preprocessed_data_roots, rows_per_frame):
    print("--- Starting Data Pre-processing (Vectorized) ---")
    for i in range(len(data_roots)):
        raw_path = Path(data_roots[i])
        processed_path = Path(preprocessed_data_roots[i])
        processed_path.mkdir(exist_ok=True)

        csv_files = list(raw_path.glob('*.csv'))
        for csv_file in tqdm(csv_files, desc="Converting CSV to .npy"):
            npy_filename = processed_path / f"{csv_file.stem}_X.npy"
            if not npy_filename.exists():
                X_full_tensor, y_full_tensor = dynamic_csv_to_tensor_vectorized(path=csv_file, rows_per_frame=rows_per_frame)
                np.save(processed_path / f"{csv_file.stem}_X.npy", X_full_tensor)
                np.save(processed_path / f"{csv_file.stem}_y.npy", y_full_tensor)

    print("--- Pre-processing Complete ---")
    return None

def dynamic_csv_to_tensor_vectorized_inference(path: Path, rows_per_frame):
    df = pd.read_csv(path)

    # --------- 1) 프레임 단위 시간 / 좌표 ---------
    # 60행마다 한 프레임
    total_frames = len(df) // rows_per_frame
    df = df.iloc[:total_frames * rows_per_frame]

    # 프레임 인덱스
    frame_idx = df.index.to_numpy() // rows_per_frame

    # 각 프레임의 모터 이동 카운트(motor_position) 추출 (첫 행만 사용)
    move_counts = df.groupby(frame_idx)["motor_position"].last().to_numpy()

    # 각 프레임의 좌표(x,y,z) 계산(진행 방향은 일단 예측해보고 어디로 가는지 알아내서 MOVE_AXIS에 순서대로 기재해야 함)
    coords = np.zeros((total_frames, 3), dtype=np.float32)
    point = np.array(START_POINT, dtype=np.float32)
    prior_move_count = move_counts[0]
    move_done_count = 0
    axis_done_count = 0
    for i in range(total_frames):
        current_move_count = move_counts[i]
        if (prior_move_count!=current_move_count):
            move_axis = MOVE_AXIS[axis_done_count % len(MOVE_AXIS)]
            direction, axis = move_axis[0], move_axis[1]
            step = 1 if direction == '+' else -1
            if axis == 'x':
                point[0] += step
            elif axis == 'y':
                point[1] += step
            elif axis == 'z':
                point[2] += step
            move_done_count += 1
            if (axis == 'x' and move_done_count >= NUM_CLASSES_X - 1) or \
               (axis == 'y' and move_done_count >= NUM_CLASSES_Y - 1) or \
               (axis == 'z' and move_done_count >= NUM_CLASSES_Z - 1):
                axis_done_count += 1
                move_done_count = 0
        if (i==total_frames-1):
            if axis == 'x':
                point[0] += step
            elif axis == 'y':
                point[1] += step
            elif axis == 'z':
                point[2] += step
        coords[i] = point
        prior_move_count = current_move_count

    # --------- 2) 좌표 보간 (piecewise linear) ---------
    # x,y,z가 변화하는 지점을 잡고 그 사이 구간은 시간 기준으로 선형 보간
    # 변화 검출
    change_mask = np.any(np.diff(coords, axis=0) != 0, axis=1)
    change_points = np.concatenate([[0], np.where(change_mask)[0] + 1])
    change_points = np.unique(change_points)

    # 보간 수행
    coords_interp = np.zeros_like(coords, dtype=np.float32)
    
    for i in range(len(change_points)-1):
        s, e = change_points[i], change_points[i+1]
        # 구간의 시작 좌표와 끝 좌표
        p_s, p_e = coords[s], coords[e]
        # 구간 내 프레임 수
        num_frames_in_segment = e - s
        # np.linspace를 사용하여 x, y, z 각 축에 대해 보간
        # endpoint=True 이므로 구간의 시작점과 끝점을 모두 포함
        x_interp = np.linspace(p_s[0], p_e[0], num=num_frames_in_segment, endpoint=False)
        y_interp = np.linspace(p_s[1], p_e[1], num=num_frames_in_segment, endpoint=False)
        z_interp = np.linspace(p_s[2], p_e[2], num=num_frames_in_segment, endpoint=False)
        # 보간된 좌표를 (N, 3) 형태로 합쳐서 할당
        coords_interp[s:e] = np.vstack([x_interp, y_interp, z_interp]).T

    # 마지막 프레임의 좌표를 명시적으로 할당
    coords_interp[-1] = coords[-1]
    
    # --------- 3) 센서 비트 -> (8,30) 이미지 ---------
    # side/b0..b3 를 이미지로 매핑
    side = df["state"].map({"left":0,"right":1}).to_numpy()
    sensor = df["sample_count"].to_numpy() % 30   # 0~29
    bits = df[["b0","b1","b2","b3"]].to_numpy(dtype=np.uint8)

    height_base = side * 4
    height_idx = height_base[:,None] + np.arange(4)
    frame_idx = frame_idx[:,None]

    X = np.zeros((total_frames, 8, 30), dtype=np.uint8)
    X[frame_idx, height_idx, sensor[:,None]] = bits
    X = X[:, np.newaxis, :, :]   # 또는 X = X.reshape(total_frames, 1, 8, 30)


    return X, coords_interp

def preprocess_dynamic_data_inference(data_roots, preprocessed_data_roots, rows_per_frame):
    print("--- Starting Data Pre-processing (Vectorized) ---")
    for i in range(len(data_roots)):
        raw_path = Path(data_roots[i])
        processed_path = Path(preprocessed_data_roots[i])
        processed_path.mkdir(exist_ok=True)

        csv_files = list(raw_path.glob('*.csv'))
        for csv_file in tqdm(csv_files, desc="Converting CSV to .npy"):
            npy_filename = processed_path / f"{csv_file.stem}_X.npy"
            if not npy_filename.exists():
                X_full_tensor, y_full_tensor = dynamic_csv_to_tensor_vectorized_inference(path=csv_file, rows_per_frame=rows_per_frame)
                np.save(processed_path / f"{csv_file.stem}_X.npy", X_full_tensor)
                np.save(processed_path / f"{csv_file.stem}_y.npy", y_full_tensor)

    print("--- Pre-processing Complete ---")
    return None

"""