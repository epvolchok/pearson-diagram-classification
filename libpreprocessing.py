#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import umap
import pandas as pd
from libservice import ServiceFuncs


class FeaturesPreprocessing:

    def __init__(self, df, copy=False):

        self.models = {
            'pca': PCA(n_components=0.95, svd_solver='full'),
            'scaler': StandardScaler(),
            'umap10d': umap.UMAP(n_components=20, min_dist=0.1, metric='cosine'),
            'umap2d': umap.UMAP(n_components=2, min_dist=0.1, metric='cosine')
        }
        if copy:
            self.df = df.copy()
        else:
            self.df = df

    def preproccessing(self, df, pipe_str): #pipe_str='PCA+UMAP+..'
        models = pipe_str.lower().split('+')
        pipes = []
        for m in models:
            pipes.append((m, self.models[m]))
        if df.size:
            pipe = Pipeline(pipes)
            results = pipe.fit_transform(df)
            return results
        else:
            return None
        
    def wrapper_preprop(self, df, pipe_str):
        df_features, excluded_part = ServiceFuncs.split_into_two(df)
        processed = self.preproccessing(df_features, pipe_str)
        df_processed = pd.DataFrame(
            processed,
            columns=[f'feat_{i}' for i in range(processed.shape[1])]
        )
        df_new = pd.concat([excluded_part, df_processed], axis=1)
        return df_new