import re
import numpy as np

# Path to the log file
log_file_path = '/project/uva_cv_lab/zezhou/point-MAE-from-unreal/Point-MAE/experiments/00-mae-zeroverse/results/zeroverse/eval-20241102-0018/finetune_scan_hardest/20241102_004341.log'

# Define regex patterns for Losses and accuracy
loss_pattern = re.compile(r"Losses = \['([\d.]+)', '([\d.]+)'\]")
acc_pattern = re.compile(r"\[Validation\] EPOCH: (\d+)\s+acc = ([\d.]+)")

# Initialize lists to store parsed data
epochs = []
loss1_values = []
loss2_values = []
acc_values = []

# Parse the log file
with open(log_file_path, 'r') as log_file:
    for line in log_file:
        # Find loss values
        loss_match = loss_pattern.search(line)
        if loss_match:
            loss1, loss2 = map(float, loss_match.groups())
            loss1_values.append(loss1)
            loss2_values.append(loss2)

        # Find accuracy values
        acc_match = acc_pattern.search(line)
        if acc_match:
            epoch, acc = int(acc_match.group(1)), float(acc_match.group(2))
            epochs.append(epoch)
            acc_values.append(acc)

# Combine parsed data into a structured array
data = np.array(list(zip(epochs, loss1_values, loss2_values, acc_values)), 
                dtype=[('epoch', 'i4'), ('loss1', 'f4'), ('loss2', 'f4'), ('acc', 'f4')])

# Save to .npy file
output_path = '/project/uva_cv_lab/zezhou/point-MAE-from-unreal/Point-MAE/experiments/useless/parsed_loss/zeroverse_finetune_scan_hardest.npy'
np.save(output_path, data)