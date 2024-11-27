# Sample 3D Synthetic Objects

Install Zeroverse
-----------------

Please follow instructions from [Zeroverse](https://github.com/desaixie/zeroverse) to install blender and zeroverse environment.

Sample 3D Synthetic Objects with different complexity and amount
---------------------------
We provide the script `create_augment_sample_shapes.py` in this directory. 

**Note**: The saving logic for `.glb` files has been commented out in the script to conserve storage space. If you need `.glb` files, you can uncomment the relevant sections in the script.

## Complexity Levels Overview

This project supports 13 levels of complexity for generating 3D shapes. Each level adds incremental challenges through the use of augmentations and the number of primitives. Below is a detailed description of each level:

### **Level 1**
- Single primitive only.
- **Number of Shapes**: 1.
- No Boolean, wireframe, or height field augmentation.

---

### **Level 2**
- Multiple primitives allowed.
- **Number of Shapes**: 1.
- No Boolean, wireframe, or height field augmentation.

---

### **Level 3**
- Multiple primitives allowed.
- **Number of Shapes**: 3.
- No Boolean, wireframe, or height field augmentation.

---

### **Level 4**
- Multiple primitives allowed.
- **Number of Shapes**: 6.
- No Boolean, wireframe, or height field augmentation.

---

### **Level 5**
- Multiple primitives allowed.
- **Number of Shapes**: 9.
- No Boolean, wireframe, or height field augmentation.

---

### **Level 6**
- Multiple primitives allowed.
- **Number of Shapes**: 9.
- Height field augmentation enabled.
- No Boolean or wireframe augmentation.

---

### **Level 7**
- Multiple primitives allowed.
- **Number of Shapes**: 9.
- Boolean augmentation enabled.
- No wireframe or height field augmentation.

---

### **Level 8**
- Multiple primitives allowed.
- **Number of Shapes**: 9.
- Wireframe augmentation enabled.
- No Boolean or height field augmentation.

---

### **Level 9**
- Multiple primitives allowed.
- **Number of Shapes**: 9.
- Boolean and height field augmentations enabled.
- No wireframe augmentation.

---

### **Level 10**
- Multiple primitives allowed.
- **Number of Shapes**: 9.
- Wireframe and height field augmentations enabled.
- No Boolean augmentation.

---

### **Level 11**
- Multiple primitives allowed.
- **Number of Shapes**: 9.
- Boolean and wireframe augmentations enabled.
- No height field augmentation.

---

### **Level 12**
- Multiple primitives allowed.
- **Number of Shapes**: 9.
- Boolean, wireframe, and height field augmentations all enabled.

---

### **Level 13**
- Multiple primitives allowed.
- **Number of Shapes**: 9.
- Boolean, wireframe, and height field augmentations all enabled.
- **Overrides**:
  - Boolean probability: **1**.
  - Wireframe probability: **1**.
  - Smooth probability: **1**. (which means no height field).

---

### Additional Notes
- **Height Field Augmentation**: Controls smoothness with a probability of **0.6**.
- **Boolean and Wireframe Augmentations**: Gradually introduced in higher levels with specific probabilities.
- **Maximum Complexity (Level 13)**: Combines all augmentations with maximum probabilities.
- **CPU only**: we find this script can be used with only 1 cpu core.

### Example Usage:
```
python create_augment_sample_shapes.py --seed 200000 --output_dir /home/zc3bp/point-MAE-from-unreal/zeroverse/experiments/01-dataset-v2/results/dataset_2 --num_shapes 30 --generate_complexity 2 
```