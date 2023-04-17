import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torchvision import models, transforms as T
from torchvision.utils import make_grid
import math, random, numpy as np
from SSLConstants import *
from tqdm import tqdm
from data import SignalDatasetV2
from collections import Counter
import pyclustertend
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import lightning.pytorch as pl

# Functions
def print_shape(x: torch.Tensor) -> None:
    """For given tensor function prints (shape, dtype, all_weights)"""
    shape, dtype = x.size()[1:], x.dtype
    total = np.prod(shape)
    print(shape, dtype, total)

def set_all_seed(seed: int) -> None:
    """Set every `possible` random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def extract_features(model:nn.Module, dataset:Dataset, pin_memory, device) -> np.ndarray:
    """Gets the output of a NN model given a dataset."""
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        #shuffle=True,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=32
        )
    features = []
    
    # Create new model from the input model without the final layer.
    if not USE_VGG:
        # Use ResNet as default model
        no_fc_model = torch.nn.Sequential(*(list(model.children())[:-1]))
    else:
        # Use VGG16
        no_fc_model = models.vgg11()
        no_fc_model.features = torch.nn.Sequential(*(list(model.features.children())))
        no_fc_model.classifier = torch.nn.Sequential(*(list(model.classifier.children())[:-1]))
    
    no_fc_model.to(device)

    # we don't need gradient calculation
    with torch.no_grad():
    # switch to evaluation mode (affects Dropout, BatchNorm)
        no_fc_model.eval()

        for (images, _) in tqdm(loader, desc='extracting features', unit='batch'):
            images = images.to(device)

            #with torch.cuda.amp.autocast():
            outputs = no_fc_model(images)

            features.append(outputs.detach().cpu())
        return torch.cat(features).numpy()

def cluster(pca, kmeans, model:nn.Module, dataset, device, return_features:bool=False, pin_memory:bool=False):
    """Generate new pseudo labels for the dataset. See arXiv:1807.05520"""

    # Extract features with NN
    features = extract_features(model, dataset, pin_memory, device)

    # Remove axes of length one since the output of the AvgPoolLayer is 4 dimensional.
    features = np.squeeze(features)

    # Remove possible NaN and inf values.
    features[~np.isfinite(features)] = 0

    # Reduce feature space with PCA
    reduced = pca.fit_transform(features)

    # L2 normalization
    norm = np.linalg.norm(reduced, axis=1)
    reduced = reduced / norm[:, np.newaxis]

    # Perform clustering/pseudo-labeling with KMeans
    # Remove possible NaN and inf values.
    reduced[~np.isfinite(reduced)] = 0
    
    pseudo_labels = list(kmeans.fit_predict(reduced))
    if return_features:
        return pseudo_labels, features, reduced
  
    return pseudo_labels


def show_cluster(cluster_centers, reduced, cluster:int, 
                 labels:list, dataset:Dataset, limit:int=32, 
                 normalize:bool=False, cmap=CMAP):
  
    """Visualize samples from given cluster."""
    images = []
    distances = []
    selected = []
    labels = np.array(labels)
    indices = np.where(labels==cluster)[0]

    if not indices.size:
        print(f'cluster: {cluster} is empty.')
        return None

    # Select random indices of the samples from given cluster.
    for i in range(limit):
        selected.append(random.randint(0, len(indices)-1))

    for i in selected:
        # Get the indices of the images in the complete dataset.
        image, _ = dataset[indices[i]]
        # im_features = reduced[i]
        cluster_center = cluster_centers[cluster]
        eucl_dist = np.linalg.norm(reduced[indices[i]] - cluster_centers[cluster])
        distances.append(eucl_dist)
        images.append(image)

    gridded = make_grid(images, pad_value=0, normalize=normalize)
    gridded = gridded.permute(1, 2, 0) # this converts [C,H,W] --> [H,W,C]

    # output of make_grid always has 3 channels
    # here we will just average channels, but they probably have indentical 
    # values.
    gridded = torch.mean(gridded, dim=-1)
    plt.figure(figsize=(25, 5))
    appendix = ' (images are normalized, dfc-euclidean distance from cluster center, ind-sample index in cluster)' if normalize else ''
    # plt.title(f'cluster: {cluster}', fontsize=16)
    plt.imshow(gridded, cmap=CMAP, filternorm=False, interpolation='none', vmin=0, vmax=1)
    plt.axis('off')
  
    for i in range(int(limit/8)):
        for j in range(8):
            plt.text(130*j + 5, 130*i + 25,"dfc: " '{0:.7f}'.format(distances[i*8+j]) + "\n" + 
                   "ind: " + str(indices[selected[i*8 + j]]), color=(1, 0.5, 0))
    
    plt.savefig("../results/output/cluster images/" 
              + "c" + str(cluster) + ".png", bbox_inches='tight',pad_inches = 0)
            
            
def show_neighbors(dist, n_neighbors, neighbors, dataset, cluster, normalize=False,
                   cmap=CMAP):
    """The function plots samples enlisted in `neighbours` and retrieves them 
    from `dataset`."""

    images = []
    selected = []

    for n in neighbors:
        image, _ = dataset[n]
        images.append(image)
        selected.append(n)

    gridded = make_grid(images, pad_value=0, normalize=normalize)
    gridded = gridded.permute(1, 2, 0) # this converts [C,H,W] --> [H,W,C]

    # output of make_grid always has 3 channels
    # here we will just average channels, but they are probably the same.
    gridded = torch.mean(gridded, dim=-1)
    plt.figure(figsize=(50, 25))

    appendix = ' (images are normalized, numbers are Euclidean distances to the first image)' if normalize else ''
    plt.title(f'nearest neighbors for cluster: {cluster}' + appendix, size=24)
    # plt.title(f'cluster: {cluster}')
    plt.imshow(gridded, cmap=cmap, filternorm=False, interpolation='none', vmin=0, vmax=1)
    plt.axis('off')

    for i in range(int(n_neighbors/8)):
        for j in range(8):
            plt.text(130*j + 5, 130*i + 25, '{0:.7f}'.format(dist[i*8+j]) + "\n" + 
                   "ind: " + str(selected[i*8 + j]), color=(1, 0.5, 0))


def calc_cluster_stats(cluster_centers, labels, reduced_features, cluster):

    distances = []
    labels = np.array(labels)
    # Get indices of the corresponding images.
    indices = np.where(labels==cluster)[0]
    cluster_center = cluster_centers[cluster]
  
    if not indices.size:
        print(f'cluster: {cluster} is empty.')
        return None
  
    # Calculate distances for each sample to the cluster center.
    for i in indices:
        # im_features = features[i]
        eucl_dist = np.linalg.norm(reduced_features[i] - cluster_centers[cluster])
        distances.append(eucl_dist)
  
    return distances
  
    
def plot_samples_grid(raw_dataset: SignalDatasetV2, num_samples: int, cmap: str, padd: int):

    plt.rc('font', size=16)

    # Show square-divided data.
    images = []
    selected = []
    normalize = False

    for n in range(num_samples):
        image, _ = raw_dataset[n]
        images.append(image)
        selected.append(n)

    gridded = make_grid(images, nrow=8, vmin=0, vmax=1, padding=padd)
    gridded = gridded.permute(1, 2, 0) # this converts [C,H,W] --> [H,W,C]

    # output of make_grid always has 3 channels
    # here we will just average channels, but they are probably the same.
    gridded = torch.mean(gridded, dim=-1)
    plt.figure(figsize=(20, 10))
    appendix = ' (images are normalized, numbers are Euclidean distances to the first image)' if normalize else ''
    # Plot some images in order to check if everything is working.
    plt.imshow(gridded, cmap=cmap, filternorm=False, interpolation='none')
    plt.xlabel('FFT bins')
    plt.ylabel('Recordings')
    plt.xticks(range(0, 1024+128, 128+padd), labels=list(range(0, 1024+128, 128)))

    plt.axis('on')
    plt.show()
    

def cnn_predict(dataset, model, pin_memory, device):
    loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            #shuffle=True,
            pin_memory=pin_memory,
            drop_last=False,
            prefetch_factor=32
            )

    cnn_outputs = []

    model.to(device)

    with torch.no_grad():
    # switch to evaluation mode (affects Dropout, BatchNorm)
        model.eval()

        for (images, _) in tqdm(loader, desc='extracting features', unit='batch'):
            images = images.to(device)

            #with torch.cuda.amp.autocast():
            outputs = model(images)

            cnn_outputs.append(outputs.detach().cpu())

        cnn_labels = torch.cat(cnn_outputs).numpy()
        tensor_outputs = torch.cat(cnn_outputs)

    # Get the class labels
    _, cnn_labels = torch.max(tensor_outputs, 1)
    cnn_labels_array = cnn_labels.numpy()
    
    return cnn_labels_array


def plot_avg_and_hist(counts: Counter, averaged_clusters: list,
                    selected_labels: list, rnd_list):
    
    if not np.any(rnd_list):
        rnd_list = range(len(selected_labels))
    # Calculate histogram for each cluster
    all_clus_data = []
    for i in range(len(averaged_clusters)):
            single_clus_data = []
            for j in range(len(selected_labels)):
                if selected_labels[j] == i:
                    single_clus_data.append(rnd_list[j]%8)

            all_clus_data.append(single_clus_data)
 

    w = 8
    h = int(np.ceil(len(counts)/w))

    plt.rc('font', size=12)
    
    fig, axs = plt.subplots(h, w, figsize=(22, 10))

    for ax, i in zip(axs.ravel(), range(len(averaged_clusters))):
        if counts[i] > 0:
            # Iterating over the subplots.
            ax.axis('off')
            ax.set_title('Cluster: ' + str(i))
            # + ', ' + 'avg=' + str(np.round(np.average(np.squeeze(averaged_clusters[i])), 4)))
            ax.imshow(np.squeeze(averaged_clusters[i]), cmap='plasma')

            divider = make_axes_locatable(ax)
            axHist = divider.append_axes("bottom", 0.5, pad=0.1)
            axHist.hist(all_clus_data[i], bins=8, range=[0, 8], label='Histogram')

            axHist.set_xticks(range(0,9,1))
            axHist.grid()

    axs[h-1, w-1].set_axis_off()
    plt.show()
    
def plot_VAT(features, reduced, num_s, cluster_num):
    # VAT
    plt.rc('font', size=16)
    plt.figure(figsize=(7, 7))
    selected_ind = np.empty((1, num_s), int)
    vat_data = reduced
    # Select random indices of the samples from given cluster.
    for i in range(num_s):
      # selected_indices.append(random.randint(0, len(samples_indices)))
      selected_ind[0][i] = random.randint(0, features.shape[0])

    features_vat = vat_data[selected_ind[0]]

    pyclustertend.visual_assessment_of_tendency.ivat(vat_data[0:num_s, :])
    plt.show()
    plt.savefig("../results/output/vat images/" 
          + "ivat c" + str(cluster_num) + ".png", facecolor='w', bbox_inches='tight', pad_inches = 0)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

    
def get_averaged_clusters(counts, used_raw, labels):
    '''Calculate averaged spectrograms for each cluster'''
    num_classes = len(counts)
    temp_list = []
    averaged_clusters = []
    
    for i in range(NUM_CLASSES):
        averaged_clusters.append(0)

    for j in range(len(used_raw)):
        temp_class = labels[j]
        averaged_clusters[temp_class] = averaged_clusters[temp_class] + np.asarray(used_raw[j][0])

    for i in range(len(averaged_clusters)):
        if counts[i] > 0:
            averaged_clusters[i] = averaged_clusters[i]/counts[i]
            # plt.imshow(np.squeeze(averaged_clusters[i]))
            # plt.show()
    
    return averaged_clusters    

def get_cluster_centers(pseudo_labels, reduced):
    cluster_centers = []
    for c in np.unique(pseudo_labels):
        clus = reduced[pseudo_labels == c]
        clus_mean = np.mean(clus, axis=0)
        cluster_centers.append(clus_mean)
    cluster_centers = np.vstack(cluster_centers)
    return cluster_centers

    
def train_epoch(model:nn.Module, optimizer, train_dataset, device, pin_memory:bool=False):
    """This function will train for single epoch. We use AMP (automatic mixed 
    precision) for faster computation."""
    # Construct dataset loader for Torch
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=32
    )

    # Storage for tracking running loss
    running_loss = []
    # Progress bar for easier tracking of the status
    pbar = tqdm(train_loader, desc='training', unit='batch')
    model.train()
    for (images, labels) in pbar:
        # Load images and pseudo-labels
        images = images.to(device, torch.float32)
        labels = labels.to(device, torch.long)
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # with torch.cuda.amp.autocast():
        # Forward pass to get output/logits
        outputs = model(images)
        # Calculate loss
        loss = F.cross_entropy(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()
        # scaler.scale(loss).backward()

        # Update parameters
        optimizer.step()

        # scaler.step(optimizer)

        # Updates the scale for next iteration
        # scaler.update()

        # Summary info (current loss)
        running_loss.append(loss.item())

        # Calculate running loss
        total_loss = sum(running_loss) / len(running_loss)

        # Obtain learning rate
        learning_rate = get_lr(optimizer)

        # Print progress and current state of the NN training
        pbar.set_postfix(loss=total_loss, lr=learning_rate)

        # Update scheduler
        #scheduler.step(total_loss)
    
    # Return stats
    return {'loss': total_loss, 'running_loss': running_loss}


