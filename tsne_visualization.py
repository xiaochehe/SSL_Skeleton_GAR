from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

def scatter(feature, labels):
    """
    docstring
    """
    # palette = np.array(sns.color_palette("hls", 10))
    palette = np.array(sns.color_palette())

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    # f = plt.figure()
    # ax = plt.subplot(111)
    
    # ac = ax.scatter(feature[:, 0], feature[:, 1], lw=0, s=40, c=palette[labels.astype(np.int)])
    for i in range(len(labels)):
        if labels[i] == 0:
            s0 = ax.scatter(feature[i, 0], feature[i, 1], lw=0, s=40, c=palette[labels[i].astype(np.int)])
        if labels[i] == 1:
            s1 = ax.scatter(feature[i, 0], feature[i, 1], lw=0, s=40, c=palette[labels[i].astype(np.int)])
        if labels[i] == 2:
            s2 = ax.scatter(feature[i, 0], feature[i, 1], lw=0, s=40, c=palette[labels[i].astype(np.int)])
        if labels[i] == 3:
            s3 = ax.scatter(feature[i, 0], feature[i, 1], lw=0, s=40, c=palette[labels[i].astype(np.int)])



    plt.xlim(feature[:, 0].min(), feature[:, 0].max())
    plt.ylim(feature[:, 1].min(), feature[:, 1].max())
    plt.legend((s0, s1, s2, s3),('moving', 'waiting', 'queueing', 'talking'), loc='best' )
    ax.axis('off')
    ax.axis('tight')



tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
feature_embed = np.load('feature.npy')
label = np.load('label.npy')

feature_embed = feature_embed
label = label
print('feature', feature_embed.shape)
print('label', label.shape)

low_dim_embed = tsne.fit_transform(feature_embed)

scatter(low_dim_embed, label)
plt.show()

