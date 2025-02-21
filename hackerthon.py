import os
import time
import numpy as np
import itertools
from concurrent.futures import ThreadPoolExecutor

def calcLoc(H, anch_pos, bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num):
    loc_result = np.zeros([H.shape[0], 2], 'float')  # 计算得到的定位结果
    return loc_result

def read_cfg_file(file_path):
    # 逐行读取文件，避免内存占用过高
    with open(file_path, 'r') as file:
        lines = file.readlines()
        info = [line.rstrip('\n').split(' ') for line in lines]
    bs_pos = list(map(float, info[0]))  # 直接转换为浮点数
    tol_samp_num, anch_samp_num, port_num, ant_num, sc_num = map(int, [info[1][0], info[2][0], info[3][0], info[4][0], info[5][0]])
    return bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num

def read_anch_file(file_path, anch_samp_num):
    anch_pos = []
    with open(file_path, 'r') as file:
        # 逐行读取并处理
        for line in file:
            line = line.rstrip('\n').split(' ')
            anch_pos.append([int(line[0]), float(line[1]), float(line[2])])
    return np.array(anch_pos)

def read_slice_of_file(file_path, start, end):
    # 使用itertools.islice来逐行读取特定范围的行
    with open(file_path, 'r') as file:
        return list(itertools.islice(file, start, end))

def process_slice_lines(slice_lines, slice_samp_num, sc_num, ant_num, port_num):
    valid_lines = []
    expected_columns = 2 * sc_num * ant_num * port_num
    for line in slice_lines:
        values = line.strip().split()
        if len(values) == expected_columns:
            valid_lines.append(values)
        else:
            print(f"Skipping line with unexpected column count: {len(values)}")
    
    if len(valid_lines) == slice_samp_num:
        Htmp = np.array(valid_lines, dtype=np.float32)
        Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))
        Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
        Htmp = np.transpose(Htmp, (0, 3, 2, 1))
        return Htmp
    else:
        raise ValueError("Insufficient valid lines in the slice to match slice_samp_num.")

def process_dataset(i, cfg_files, pos_files, data_files, output_files):
    print(f'Processing Dataset {i + 1}')
    
    cfg_path = cfg_files[i]
    bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num = read_cfg_file(cfg_path)
                
    anch_pos_path = pos_files[i]
    anch_pos = read_anch_file(anch_pos_path, anch_samp_num)

    data_path = data_files[i]
    slice_samp_num = 1000  
    slice_num = int(tol_samp_num / slice_samp_num)  
    
    output_path = output_files[i]
    with open(output_path, 'w') as output_file:
        for slice_idx in range(slice_num):  
            print(f'Processing slice {slice_idx + 1}/{slice_num}')
            slice_start = slice_idx * slice_samp_num
            slice_end = (slice_idx + 1) * slice_samp_num

            slice_lines = read_slice_of_file(data_path, slice_start, slice_end)
            try:
                Htmp = process_slice_lines(slice_lines, slice_samp_num, sc_num, ant_num, port_num)
            except ValueError as e:
                print(e)
                continue  # 跳过当前分片的处理

            result = calcLoc(Htmp, anch_pos, bs_pos, slice_samp_num, anch_samp_num, port_num, ant_num, sc_num)

            # 预处理锚点位置索引，避免重复计算
            anch_pos_dict = {int(anch[0] - 1): anch[1:] for anch in anch_pos}

            for idx in range(anch_samp_num):
                global_rowIdx = int(anch_pos[idx][0] - 1)  # 锚点在整个数据集的索引
                if slice_start <= global_rowIdx < slice_end:  
                    local_rowIdx = global_rowIdx - slice_start  
                    result[local_rowIdx] = np.array([anch_pos_dict[global_rowIdx][0], anch_pos_dict[global_rowIdx][1]])

            np.savetxt(output_file, result, fmt='%.4f %.4f')

    print(f'Output for Dataset {i + 1} saved to: {output_path}')


if __name__ == "__main__":
    print("<<< Welcome to 2024 Wireless Algorithm Contest! >>>\n")
    
    cfg_files = [
        "D:/HuaWei/data1/Dataset1CfgData1.txt",
        "D:/HuaWei/data1/Dataset1CfgData2.txt",
        "D:/HuaWei/data1/Dataset1CfgData3.txt",
    ]
    
    pos_files = [
        "D:/HuaWei/data1/Dataset1InputPos1.txt",
        "D:/HuaWei/data1/Dataset1InputPos2.txt",
        "D:/HuaWei/data1/Dataset1InputPos3.txt",
    ]
    
    data_files = [
        "D:/HuaWei/data1/Dataset1InputData1.txt",
        "D:/HuaWei/data1/Dataset1InputData2.txt",
        "D:/HuaWei/data1/Dataset1InputData3.txt",
    ]
    
    output_files = [
        "D:/HuaWei/data1/Dataset1OutputData1.txt",
        "D:/HuaWei/data1/Dataset1OutputData2.txt",
        "D:/HuaWei/data1/Dataset1OutputData3.txt"
    ]



#1, 读取文件效率的优化：在read_cfg_file、read_anch_file 和 read_slice_of_file 中，使用了 readlines 来一次性读取整个文件，这在处理大文件时效率较低。我们可以通过逐行读取，避免一次性读取整个文件，提高内存使用效率。
#2. 数组操作的优化：在process_slice_lines中，先将数据转换为NumPy数组，再进行一些复杂的操作。如果能够优化数据的存储方式，直接进行矩阵运算
#3. 并行处理：Python中使用concurrent.futures或multiprocessing来并行处理每个数据集
#4. calcLoc方法可以使用更高效的定位算法。如果使用矩阵运算或者利用现有的优化库（例如scipy中的线性代数运算），定位计算的速度可以进一步提升。
#5. 在计算锚点位置时，每次都会进行遍历，检查每个锚点是否在当前分片内。可以提前预处理锚点位置的索引，避免每次都进行遍历。


'''
import os
import time
import numpy as np
import itertools
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import lstsq

# 计算定位结果的示例函数（这里可以优化定位算法）
def calcLoc(H, anch_pos, bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num):
    loc_result = np.zeros([H.shape[0], 2], 'float')  # 计算得到的定位结果
    for i in range(H.shape[0]):
        # 使用最小二乘法进行定位（这里可以根据具体情况进行优化）
        A = np.vstack([H[i].real, H[i].imag]).T
        b = np.array([anch_pos[:, 1], anch_pos[:, 2]]).T
        x, resids, rank, s = lstsq(A, b)  # 最小二乘法
        loc_result[i] = x
    return loc_result

# 读取配置文件
def read_cfg_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        info = [line.rstrip('\n').split(' ') for line in lines]
    bs_pos = list(map(float, info[0]))  # 直接转换为浮点数
    tol_samp_num, anch_samp_num, port_num, ant_num, sc_num = map(int, [info[1][0], info[2][0], info[3][0], info[4][0], info[5][0]])
    return bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num

# 读取锚点位置文件
def read_anch_file(file_path, anch_samp_num):
    anch_pos = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.rstrip('\n').split(' ')
            anch_pos.append([int(line[0]), float(line[1]), float(line[2])])
    return np.array(anch_pos)

# 读取大文件的分片数据
def read_slice_of_file(file_path, start, end):
    with open(file_path, 'r') as file:
        return list(itertools.islice(file, start, end))

# 处理读取的数据（将数据转换为NumPy数组，进行矩阵操作）
def process_slice_lines(slice_lines, slice_samp_num, sc_num, ant_num, port_num):
    valid_lines = []
    expected_columns = 2 * sc_num * ant_num * port_num
    for line in slice_lines:
        values = line.strip().split()
        if len(values) == expected_columns:
            valid_lines.append(values)
        else:
            print(f"Skipping line with unexpected column count: {len(values)}")
    
    if len(valid_lines) == slice_samp_num:
        Htmp = np.array(valid_lines, dtype=np.float32)
        Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))
        Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
        Htmp = np.transpose(Htmp, (0, 3, 2, 1))
        return Htmp
    else:
        raise ValueError("Insufficient valid lines in the slice to match slice_samp_num.")

# 预处理锚点位置（创建字典，避免每次遍历）
def preprocess_anchor_positions(anch_pos):
    return {int(anch[0] - 1): anch[1:] for anch in anch_pos}

# 处理单个数据集
def process_dataset(i, cfg_files, pos_files, data_files, output_files):
    print(f'Processing Dataset {i + 1}')
    
    cfg_path = cfg_files[i]
    bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num = read_cfg_file(cfg_path)
                
    anch_pos_path = pos_files[i]
    anch_pos = read_anch_file(anch_pos_path, anch_samp_num)
    anch_pos_dict = preprocess_anchor_positions(anch_pos)

    data_path = data_files[i]
    slice_samp_num = 1000  
    slice_num = int(tol_samp_num / slice_samp_num)  
    
    output_path = output_files[i]
    with open(output_path, 'w') as output_file:
        for slice_idx in range(slice_num):  
            print(f'Processing slice {slice_idx + 1}/{slice_num}')
            slice_start = slice_idx * slice_samp_num
            slice_end = (slice_idx + 1) * slice_samp_num

            slice_lines = read_slice_of_file(data_path, slice_start, slice_end)
            try:
                Htmp = process_slice_lines(slice_lines, slice_samp_num, sc_num, ant_num, port_num)
            except ValueError as e:
                print(e)
                continue  # 跳过当前分片的处理

            result = calcLoc(Htmp, anch_pos, bs_pos, slice_samp_num, anch_samp_num, port_num, ant_num, sc_num)

            for idx in range(anch_samp_num):
                global_rowIdx = int(anch_pos[idx][0] - 1)  # 锚点在整个数据集的索引
                if slice_start <= global_rowIdx < slice_end:  
                    local_rowIdx = global_rowIdx - slice_start  
                    result[local_rowIdx] = anch_pos_dict[global_rowIdx]

            np.savetxt(output_file, result, fmt='%.4f %.4f')

    print(f'Output for Dataset {i + 1} saved to: {output_path}')

# 并行处理多个数据集
def process_dataset_parallel(cfg_files, pos_files, data_files, output_files):
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(cfg_files)):
            futures.append(executor.submit(process_dataset, i, cfg_files, pos_files, data_files, output_files))
        
        for future in futures:
            future.result()  # 处理异常并确保任务已完成

if __name__ == "__main__":
    print("<<< Welcome to 2024 Wireless Algorithm Contest! >>>\n")
    
    cfg_files = [
        "D:/HuaWei/data1/Dataset1CfgData1.txt",
        "D:/HuaWei/data1/Dataset1CfgData2.txt",
        "D:/HuaWei/data1/Dataset1CfgData3.txt",
    ]
    
    pos_files = [
        "D:/HuaWei/data1/Dataset1InputPos1.txt",
        "D:/HuaWei/data1/Dataset1InputPos2.txt",
        "D:/HuaWei/data1/Dataset1InputPos3.txt",
    ]
    
    data_files = [
        "D:/HuaWei/data1/Dataset1InputData1.txt",
        "D:/HuaWei/data1/Dataset1InputData2.txt",
        "D:/HuaWei/data1/Dataset1InputData3.txt",
    ]
    
    output_files = [
        "D:/HuaWei/data1/Dataset1OutputData1.txt",
        "D:/HuaWei/data1/Dataset1OutputData2.txt",
        "D:/HuaWei/data1/Dataset1OutputData3.txt"
    ]
    
    # 并行处理所有dataset
    process_dataset_parallel(cfg_files, pos_files, data_files, output_files)
'''