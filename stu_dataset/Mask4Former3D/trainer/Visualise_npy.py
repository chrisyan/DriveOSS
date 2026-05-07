import matplotlib.pyplot as plt
import numpy as np
import torch

logit = torch.tensor ([
    [0.9,0.1,0.0],
    [0.3,0.6,0.1]
], device='cuda')

mask = torch.tensor ([
    [1.0,-5.0],
    [-10.,-10.]
], device='cuda')


inv_map = torch.tensor ([1,1])
print("inv shape_0:",inv_map.shape)

confid = mask.float().sigmoid().matmul(logit)
print("confid:",confid)
max_logit = torch.max(confid, dim=1).values[inv_map]
print("max_logit:",max_logit)
print("inv shape:",inv_map.shape)



'''
#Load npy
mean_dict = np.load('', allow_pickle=True).item()  
var_dict = np.load('', allow_pickle=True).item()
mask_mean_dict = np.load('', allow_pickle=True).item()

#print("Mean logits per class:", mean_dict)
#print("Variance per class:", var_dict)
#print("Mean mask probabilities per class:", mask_mean_dict)


#Visualisation : Bar plot class wise comparison
# Extract class IDs and values
classes = sorted(mean_dict.keys())
mean_values = [mean_dict[c] for c in classes]
var_values = [var_dict[c] for c in classes]
mask_values = [mask_mean_dict[c] for c in classes]

# Plot mean logits
plt.figure(figsize=(10, 5))
plt.bar(classes, mean_values, color='skyblue')
plt.xlabel('Class ID')
plt.ylabel('Mean Logit')
plt.title('Mean Logit Scores per Class')
plt.xticks(classes)
plt.grid(axis='y', linestyle='--')
plt.show()

# Plot variance
plt.figure(figsize=(10, 5))
plt.bar(classes, var_values, color='salmon')
plt.xlabel('Class ID')
plt.ylabel('Variance')
plt.title('Logit Variance per Class')
plt.xticks(classes)
plt.grid(axis='y', linestyle='--')
plt.show()

# Plot mask probabilities
plt.figure(figsize=(10, 5))
plt.bar(classes, mask_values, color='lightgreen')
plt.xlabel('Class ID')
plt.ylabel('Mean Mask Probability')
plt.title('Average Mask Probability per Class')
plt.xticks(classes)
plt.grid(axis='y', linestyle='--')
plt.show()


#Visualisation : Scatter Mean vs variance

plt.figure(figsize=(8, 6))
plt.scatter(mean_values, var_values, c=classes, cmap='viridis')
plt.colorbar(label='Class ID')
plt.xlabel('Mean Logit')
plt.ylabel('Variance')
plt.title('Mean Logit vs. Variance per Class')
plt.grid(True)
plt.show()
'''