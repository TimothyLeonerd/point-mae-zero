import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

dataroot = "/project/uva_cv_lab/zezhou/point-MAE-from-unreal/zeroverse/experiments/01-dataset-v3/results/dataset_12"
out_filepath = "/project/uva_cv_lab/zezhou/point-MAE-from-unreal/Point-MAE/data/ZeroVerse/split_complexity_level_12_2000"

os.makedirs(out_filepath, exist_ok=True,)
# Number of worker threads
num_workers = 32
# Specify the maximum number of shapes to process
max_shapes = 998  # Set this to the number of shapes you want to process

# Function to check if file exists
def check_file(shape_dir):
    file_path = os.path.join(dataroot, shape_dir, "object_aug.npy")
    if os.path.exists(file_path):
        return os.path.join(shape_dir, "object_aug.npy")
    else:
        print(f"File {file_path} does not exist")
        return None

# Get list of all shapes in parallel with tqdm progress bar, limited by max_shapes
shape_list = os.listdir(dataroot)[:max_shapes]
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    all_shapes = list(filter(None, tqdm(executor.map(check_file, shape_list), total=len(shape_list), desc="Processing files")))

# Split data into train and test sets
train_shapes = all_shapes[:100]
test_shapes = all_shapes[100:995]  # Adjusted to 200 test shapes after train shapes

# Write the lists to files
os.makedirs(out_filepath, exist_ok=True)
with open(os.path.join(out_filepath, "train.txt"), "w") as f:
    f.writelines(f"{shape}\n" for shape in train_shapes)

with open(os.path.join(out_filepath, "test.txt"), "w") as f:
    f.writelines(f"{shape}\n" for shape in test_shapes)