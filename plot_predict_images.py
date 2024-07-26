import matplotlib.pyplot as plt

def plot_multiple_images(original, true_mask, pred_mask, start_index, end_index):
    for i in range(start_index, end_index + 1):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original[i])
        axes[0].set_title(f'Original Image {i+1}')
        axes[0].axis('off')

        axes[1].imshow(true_mask[i], cmap='gray')
        axes[1].set_title(f'True Mask {i+1}')
        axes[1].axis('off')

        axes[2].imshow(pred_mask[i] > 0.5, cmap='gray')  
        axes[2].set_title(f'Predicted Mask {i+1}')
        axes[2].axis('off')

        plt.show()
