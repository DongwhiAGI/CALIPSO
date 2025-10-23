import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

def stem_to_xyz(stem: str):
    z_str, y_str, x_str = stem.split('_')
    x, y, z = int(x_str), int(y_str), int(z_str)
    return x, y, z

def load_static_data(preprocessed_data_roots, val_ratio, test_ratio):
    X_train_list = []
    y_train_list = []
    X_validation_list = []
    y_validation_list = []
    X_test_list = []
    y_test_list = []
    for i in range(len(preprocessed_data_roots)):
        root_path = Path(preprocessed_data_roots[i])
        all_files = sorted(root_path.glob('*.npy'))
        if not all_files:
            print(f"Error: No .npy files found in '{preprocessed_data_roots[i]}'.")
            return
        for f in tqdm(all_files, desc="Generating and Splitting Clips"):
            arr = np.load(f, mmap_mode="r")         # 빠르고 메모리 친화적
            if arr.shape[1:] == (1, 8, 30):
                # (T, 1, 8, 30)
                x_np = arr.copy()
            else:
                raise ValueError(f"Unexpected shape {arr.shape}, expected (T,1,8,30)")
            N = x_np.shape[0]
            validation_sample_len = round(N * val_ratio)
            test_sample_len  = round(N * test_ratio)
            train_sample_len = N - validation_sample_len - test_sample_len
            
            x, y, z = stem_to_xyz(f.stem)
            coords = np.tile([x, y, z], (arr.shape[0], 1))
            X_train_list.append(x_np[:train_sample_len])
            X_validation_list.append(x_np[train_sample_len:train_sample_len+validation_sample_len])
            X_test_list.append(x_np[train_sample_len+validation_sample_len:])
            y_train_list.append(coords[:train_sample_len])
            y_validation_list.append(coords[train_sample_len:train_sample_len+validation_sample_len])
            y_test_list.append(coords[train_sample_len+validation_sample_len:])

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_valid = np.concatenate(X_validation_list, axis=0)
    y_valid = np.concatenate(y_validation_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    print(f'load static data: train({len(y_train)}), validation({len(y_valid)}), test({len(y_test)})')
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def load_static_data_connected(preprocessed_data_roots):
    X_data_list = []
    y_data_list = []
    for i in range(len(preprocessed_data_roots)):
        root_path = Path(preprocessed_data_roots[i])
        all_files = sorted(root_path.glob('*.npy'))
        if not all_files:
            print(f"Error: No .npy files found in '{preprocessed_data_roots[i]}'.")
            return
        for f in tqdm(all_files, desc="Generating and Splitting Clips"):
            arr = np.load(f, mmap_mode="r")         # 빠르고 메모리 친화적
            if arr.shape[1:] == (1, 8, 30):
                # (T, 1, 8, 30)
                x_np = arr.copy()
            else:
                raise ValueError(f"Unexpected shape {arr.shape}, expected (T,1,8,30)")        
            
            x, y, z = stem_to_xyz(f.stem)
            coords = np.tile([x, y, z], (arr.shape[0], 1))
            X_data_list.append(x_np)
            y_data_list.append(coords)

    X_data = np.concatenate(X_data_list, axis=0)
    y_data = np.concatenate(y_data_list, axis=0)
    print(f'load static data: total({len(y_data)})')
    
    return X_data, y_data

def filter_and_remap_coords(X, y, x_range=None, y_range=None, z_range=None):
    # --- 수정된 경고 기능: 입력된 range 길이 비교 ---
    range_lengths = {}
    if x_range:
        range_lengths['X'] = x_range[1] - x_range[0]
    if y_range:
        range_lengths['Y'] = y_range[1] - y_range[0]
    if z_range:
        range_lengths['Z'] = z_range[1] - z_range[0]

    # 제공된 범위가 2개 이상이고, 그 길이가 서로 다를 경우 경고 메시지를 출력합니다.
    if len(range_lengths) > 1:
        # set을 사용하여 고유한 길이 값만 남깁니다. 길이가 1보다 크면 서로 다른 길이가 존재한다는 의미입니다.
        if len(set(range_lengths.values())) > 1:
            details = ", ".join([f"{axis}: {length}" for axis, length in range_lengths.items()])
            print(f"⚠️ Warning: The input range lengths are not equal.")
            print(f"   Provided range lengths -> {details}")
    # --- 여기까지 수정된 부분 ---
    
    if y.shape[1] != 3:
        raise ValueError("y array must have 3 columns for x, y, z coordinates.")

    # 1. 데이터 필터링 (Boolean Mask 생성)
    mask = np.ones(len(y), dtype=bool) # 모든 데이터가 True인 마스크로 시작

    if x_range:
        mask &= (y[:, 0] >= x_range[0]) & (y[:, 0] <= x_range[1])
    if y_range:
        mask &= (y[:, 1] >= y_range[0]) & (y[:, 1] <= y_range[1])
    if z_range:
        mask &= (y[:, 2] >= z_range[0]) & (y[:, 2] <= z_range[1])

    # 필터링된 데이터 추출
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    if len(y_filtered) == 0:
        print("Warning: No data left after filtering.")
        return X_filtered, y_filtered
    
    # 2. 좌표 리매핑
    y_remapped = np.zeros_like(y_filtered)
    
    # 각 축(x, y, z)에 대해 리매핑 수행
    for axis in range(3):
        # 현재 축의 고유한 좌표 값들을 오름차순으로 찾음
        unique_coords = np.unique(y_filtered[:, axis])
        
        # searchsorted를 사용하여 기존 좌표 값을 새로운 0-based 인덱스로 변환
        # 예: unique_coords가 [10, 15, 20] 이라면,
        # 10 -> 0, 15 -> 1, 20 -> 2 로 매핑됨
        y_remapped[:, axis] = np.searchsorted(unique_coords, y_filtered[:, axis])
        
    return X_filtered, y_remapped

def load_dynamic_data(preprocessed_data_roots, val_ratio, test_ratio):
    X_train_list = []
    y_train_list = []
    X_validation_list = []
    y_validation_list = []
    X_test_list = []
    y_test_list = []
    for i in range(len(preprocessed_data_roots)):
        root_path = Path(preprocessed_data_roots[i])
        all_files = sorted(root_path.glob('*.npy'))
        if not all_files:
            print(f"Error: No .npy files found in '{preprocessed_data_roots[i]}'.")
            return

        groups = defaultdict(dict)
        for f in all_files:
            name = f.stem  # 'dataname0_X' 같은 파일명에서 확장자 제거
            if name.endswith("_X"):
                groups[name[:-2]]["X"] = f
            elif name.endswith("_y"):
                groups[name[:-2]]["y"] = f
        for dataname, files in tqdm(groups.items(), desc="Loading paired data"):
            if "X" not in files or "y" not in files:
                raise ValueError(f"{dataname}에 X 또는 y 파일이 없음")
            X_arr = np.load(files["X"], mmap_mode="r")
            y_arr = np.load(files["y"], mmap_mode="r")
            if X_arr.shape[1:] != (1, 8, 30):
                raise ValueError(f"{files['X']} shape={X_arr.shape}, expected (T,1,8,30)")
            if y_arr.shape[1] != 3:
                raise ValueError(f"{files['y']} shape={y_arr.shape}, expected (T,3)")
            train_sample_len = int(X_arr.shape[0] * (1-(val_ratio+test_ratio))) 
            validation_sample_len = int(X_arr.shape[0] * val_ratio)
            test_sample_len = int(X_arr.shape[0] * test_ratio)
            
            X_train_list.append(X_arr[:train_sample_len])
            X_validation_list.append(X_arr[train_sample_len:train_sample_len+validation_sample_len])
            X_test_list.append(X_arr[train_sample_len+validation_sample_len:])
            y_train_list.append(y_arr[:train_sample_len])
            y_validation_list.append(y_arr[train_sample_len:train_sample_len+validation_sample_len])
            y_test_list.append(y_arr[train_sample_len+validation_sample_len:])
    
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_valid = np.concatenate(X_validation_list, axis=0)
    y_valid = np.concatenate(y_validation_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    print(f'load static data: train({len(y_train)}), validation({len(y_valid)}), test({len(y_test)})')

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def load_dynamic_data_inference(preprocessed_data_roots):
    X_data_list = []
    y_data_list = []
    for i in range(len(preprocessed_data_roots)):
        root_path = Path(preprocessed_data_roots[i])
        all_files = sorted(root_path.glob('*.npy'))
        if not all_files:
            print(f"Error: No .npy files found in '{preprocessed_data_roots[i]}'.")
            return

        groups = defaultdict(dict)
        for f in all_files:
            name = f.stem  # 'dataname0_X' 같은 파일명에서 확장자 제거
            if name.endswith("_X"):
                groups[name[:-2]]["X"] = f
            elif name.endswith("_y"):
                groups[name[:-2]]["y"] = f
        for dataname, files in tqdm(groups.items(), desc="Loading paired data"):
            if "X" not in files or "y" not in files:
                raise ValueError(f"{dataname}에 X 또는 y 파일이 없음")
            X_arr = np.load(files["X"], mmap_mode="r")
            y_arr = np.load(files["y"], mmap_mode="r")
            if X_arr.shape[1:] != (1, 8, 30):
                raise ValueError(f"{files['X']} shape={X_arr.shape}, expected (T,1,8,30)")
            if y_arr.shape[1] != 3:
                raise ValueError(f"{files['y']} shape={y_arr.shape}, expected (T,3)")
            
            X_data_list.append(X_arr)
            y_data_list.append(y_arr)
    
    X_data = np.concatenate(X_data_list, axis=0)
    y_data = np.concatenate(y_data_list, axis=0)

    return X_data, y_data

def load_inference_data(preprocessed_data_roots):
    X_data_list = []
    for i in range(len(preprocessed_data_roots)):
        root_path = Path(preprocessed_data_roots[i])
        all_files = sorted(root_path.glob('*.npy'))
        if not all_files:
            print(f"Error: No .npy files found in '{preprocessed_data_roots[i]}'.")
            return
        for f in tqdm(all_files, desc="Generating and Splitting Clips"):
            arr = np.load(f, mmap_mode="r")         # 빠르고 메모리 친화적
            if arr.shape[1:] == (1, 8, 30):
                # (T, 1, 8, 30)
                x_np = arr.copy()
            else:
                raise ValueError(f"Unexpected shape {arr.shape}, expected (T,1,8,30)")        
            X_data_list.append(x_np)
    X_data = np.concatenate(X_data_list, axis=0)

    return X_data