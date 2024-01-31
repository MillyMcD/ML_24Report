import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

class UMAPWrapper:
    """
    For training a UMAP dimensionality reduction algorithm
    """
    def __init__(self,neighbours=10,distance=0.2,dimension:int=2,seed:int=42,metric='euclidean'):
        self.neighbours=neighbours
        self.distance = distance
        self.dimension=dimension
        self.seed=seed
        self.metric=metric
        self.fitted=False

        self.mapper = UMAP(n_neighbors=self.neighbours,
                           min_dist=self.distance,
                           n_components=dimension,
                           random_state = self.seed,
                           transform_seed = self.seed,
                           metric=self.metric)

    def train(self,train_x):
        self.mapper.fit(train_x)
        self.fitted=True

    def project(self,vectors):
        if self.fitted is False:
            print('Need to fit first')
            return
        return self.mapper.transform(vectors)

class DBSCANWrapper:
    """
    Training a DBSCAN model. Also adds a predict method
    """
    def __init__(self,sample_size:int,eps:float=0.5,min_samples:int=None,metric='euclidean'):

        self.min_samples = min_samples or 5

        self.model = DBSCAN(min_samples = self.min_samples,
                            eps=eps,
                            n_jobs=1,
                            metric=metric)

    def train(self,train_x):
        self.model.fit(train_x)

    def predict(self,vectors):
        offsets = self.model.components_[:,np.newaxis] - vectors[np.newaxis,:]
        distance = np.linalg.norm(offsets,axis=2)

        min_dist = np.min(distance,axis=0)
        amin_dist = np.argmin(distance,axis=0)
        closest  = np.where(min_dist < self.model.eps,amin_dist,-1)

        core_labels = np.hstack([self.model.core_sample_indices_,[-1]])
        model_labels = np.hstack([self.model.labels_,[-1]])
        labels = model_labels[core_labels[closest]]
        return labels

def get_eps(data):
    """
    determine eps for dbscan
    """
    model = NearestNeighbors(n_neighbors=2)
    neighbours = model.fit(data)
    distances, indices = neighbours.kneighbors(data)
    
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    fig,ax = plt.subplots()
    ax.plot(distances)
    ax.set(title='DBSCAN Eps vs step',xlabel='steps',ylabel='EPS')
    ax.grid()

def train_projectors(all_data,dimension=20,neighbours=10):
    """train umap projectors"""
    projector =UMAPWrapper(dimension=dimension,neighbours=neighbours)
    proj_viz  =UMAPWrapper(dimension=2,neighbours=neighbours)

    projector.train(all_data)
    proj_viz.train(all_data)

    vec_proj = projector.project(all_data)
    vec_viz  = proj_viz.project(all_data)
    return vec_proj, vec_viz

def train_dbscan_and_get_labels(proj_data,eps):
    """train dbscan"""
    dbscan = DBSCANWrapper(sample_size=len(proj_data),eps=eps)
    dbscan.train(proj_data)
    return dbscan.predict(proj_data)

def get_most_similar(idx,vector,data,n=5):
    """get top N most similar vectors"""
    pw = pairwise_distances(vector)
    print('Track:###',data.iloc[idx][['Track Name','Artist Name(s)','Artist Genres','Album Name']])
    print('top 5 most similar....')
    distances = np.argsort(pw[idx])
    for i in distances[1:n]:
        print('-----')
        print(data.iloc[i][['Track Name','Artist Name(s)','Artist Genres','Album Name']])
        print('Distance:',pw[idx][i])

def plot_clusters(vec_viz,labels):
    """plot clusters"""
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=-1,vmax=len(np.unique(labels)))
    cmap = mpl.cm.nipy_spectral
    col_list = [cmap(norm(i)) for i in labels]
    fig,ax = plt.subplots()
    ax.scatter(*vec_viz.T,c=col_list)