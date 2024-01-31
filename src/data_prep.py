import pandas as pd
import numpy as np
import sent2vec
from sklearn.decomposition import PCA
from collections import Counter
from umap import UMAP

class DataPrep:
    """
    Data Prep class. Formats the Spotify dataset into ML format
    """
    def __init__(self,df,sent2vec_weights='model.bin'):
        self.df=df
        self.sent2vec_model=sent2vec.Sent2vecModel()
        self.sent2vec_model.load_model(sent2vec_weights)

        self.orig_df = self.df.copy()
        

    def drop_columns(self,columns):
        """
        Drop unneeded columns
        """
        self.df=self.df.drop(columns=[i for i in columns if i in self.df.columns])

    def prepare_datetimes(self,add_decade=True,date_column='Album Release Date'):
        """
        Prepare the datetimes/fix issues. Add decades if reequested
        """
        if not date_column in self.df:
            return
            
        dates = []
        #looping through all the dates to catch edge cases`
        for j,i in enumerate(self.df[date_column]):
            #print(j,i)
            if str(i) == 'nan-01-01':
                dates.append('2000-01-01')
            elif str(i) == '0000-01-01':
                dates.append('2000-01-01')
            elif i == '0000':
                dates.append('2000-01-01')
            elif len(str(i)) == 4:
                dates.append(f'{i}-01-01')
            elif len(str(i)) == 7:
                dates.append(f'{i}-01')
            else:
                dates.append(i)
  
        #set the 'cleaned' dates   
        self.df[date_column] = dates
        #converting string to datetime type
        self.df[date_column] = pd.to_datetime(self.df[date_column])

        #adding decades if requested
        if add_decade:
            decades=[]
            for i in (self.df[date_column]): 
                i=i.to_pydatetime()
                decade=str(i.year)[:3]+'0'
                decades.append(decade)
            self.df['Decade']=decades

    def encode_categoricals(self,columns):
        """
        This function creates a mapping from unique entry to unique number for all categorical variables
        """
        self.encodings = {}
        #Loop through each categorical column
        for column in columns:
            if not column in self.df.columns:
                continue

            #Using value counts to get all unique items, and enumerate to map them into intergers 
            self.encodings[column] = {k:i for i,(k,v) in enumerate(self.df[column].value_counts().to_dict().items())}
            #Now using the map function to replace unique items with unique numbers
            self.df[column] = self.df[column].map(self.encodings[column])

    def encode_datetime(self,date_column='Album Release Date'):
        """
        Encode datetime, so earliest is 0 and latest is the number of days from earliest
        """
        if not date_column in self.df.columns:
            return
        #calculated the minimum time in the dataset
        min_time=self.df[date_column].min()
        #subtract minimum time from every datetime, then convert column
        #to the number of days from the earliest date
        self.df[date_column]=(self.df[date_column]-min_time).dt.days

    @staticmethod
    def pull_out_items(df, column, separator=','):
        """
        Static method to pull all fields from a row/column in CSV style such as genre
        and artist album names
        """
        names=[]
        for i in df[column].values:
            i = str(i).strip()
            if not separator in i:
                names.append(i)
            else :
                nme=i.split(separator)
                names.extend(nme)
        return names
    
    def encode_genre(self,genre_column='Artist Genres',topk=30):
        """
        Encode the genre. Simplify a number of more obscure categories whilst retaining the top30 occuring as unqiue
        genres. Then create a one-hot-encoding (i.e N columns) representation.
        """
        if not genre_column in self.df.columns:
            return
        
        #filling missing values with 'unknown'
        self.df[genre_column] = self.df[genre_column].fillna('unknown')
        #pulling out every single genre
        genres=self.pull_out_items(self.df,genre_column)

        #use a counter to get counts for each genre
        genre_count =Counter(genres)
        genres,counts = zip(*genre_count.most_common())

        #Use custom grouped genres
        cats= ['pop','rock','r&b','techno','metal','dance','rap','indie','hip hop','soul','folk','disco','country']

        #loop through each genre after topk and map specific genre to grouped genre
        mapping = {}
        for genre_name in genres[topk:]:
            for grouped_genre in cats:
                if grouped_genre in genre_name:
                    mapping[genre_name]=grouped_genre
                    continue
            #if not present, set as other
            if genre_name not in mapping:
                mapping[genre_name]='other'

        #treat topk as standard genres
        for genre_name in genres[:topk]:
            mapping[genre_name]=genre_name

        #create mapping from integer to genre
        all_cats = set(mapping.values())
        int_map  = {k:i for i,k in enumerate(all_cats)}
        map_int =  {i:k for i,k in enumerate(all_cats)}

        #loop through each genre row, split, and map to one-hot encoding
        #depending on how many genres appear in it
        per_track_genres = []
        for i in self.df[genre_column]:
            encoding = np.zeros(len(int_map))
            if ',' in i:
                for j in i.split(','):
                    idx = int_map[mapping[j]]
                    encoding[idx]=1
            else:
                idx = int_map[mapping[i]]
                encoding[idx] = 1
            
            per_track_genres.append(encoding)

        #stack the one hot encodings, then add as a column
        one_hots = np.stack(per_track_genres).T
        for i,genre in enumerate(one_hots):
            self.df[f'Genre {map_int[i]}'] = genre

        #drop the genre column
        self.df = self.df.drop(columns=genre_column)

    def encode_album_artists(self,album_artists_column='Album Artist Name(s)'):
        """
        Encode all album artists. Determines max seen on one album (N) and then
        adds this number of columns, where each column has a cell containing the
        unique artist mapping. For rows with less than < N feature artists,
        the values are set to -1.
        """
        if not album_artists_column in self.df.columns:
            return
        
        #firstly, drop nans
        self.df = self.df.loc[~self.df[album_artists_column].isna()]

        #convert to string
        self.df[album_artists_column] = self.df[album_artists_column].astype('str')

        #strip/lowercase
        self.df[album_artists_column] = [i.strip().lower() for i in self.df[album_artists_column]]

        #grab all unique artists
        artists = self.pull_out_items(self.df,album_artists_column)
        artist_count =Counter(artists)

        #map artists to integer
        int_map = {k:i for i,(k,v) in enumerate(artist_count.most_common())}

        #grab all album artist. Work out max number seen on one track in dataset
        all_artists    = [i for i in self.df[album_artists_column]]
        max_num_artist = max([len(str(i).split(',')) for i in all_artists])

        per_track_artists = []
        for i in self.df[album_artists_column]:
            encoding = np.ones((max_num_artist))*-1
            if ',' in i:
                for j,artist in enumerate(i.split(',')):
                    encoding[j] = int_map[artist]
            else:
                encoding[0] = int_map[i]
            per_track_artists.append(encoding)

        vectors = np.stack(per_track_artists).T
        for i,artist in enumerate(vectors):
            self.df[f'Album Artist {i}'] = artist

        self.df = self.df.drop(columns=album_artists_column)

    def set_dtypes(self):
        """
        Set dtypes of all non-object columns to float32
        """
        non_obj_cols = [i for i in self.df.select_dtypes([np.number,bool]).columns]
        type_mapping = {i:np.float32 for i in non_obj_cols}
        self.df = self.df.astype(type_mapping)
        

    def encode_text(self,text_column,n_components=2,method='pca'):
        """
        Encode text via sent-2-vec and then use PCA to reduce components
        """
        names=self.df[text_column].values
        names=[str(i) for i in names]
        emb=self.sent2vec_model.embed_sentences(names)
        
        if n_components < 600:
            print('Using PCA')
            pca=PCA(n_components=n_components)
            emb_pca=pca.fit_transform(emb)
        else:
            emb_pca = emb

        for i,component in enumerate(emb_pca.T):
            self.df[f'{text_column}_pca_{i}'] = component

        self.df = self.df.drop(columns=text_column)
        

    def prepare_data(self,drop_columns,cat_columns,text_columns,topk_genres:int=30,add_decade:bool=False,n_components_text=2):
        """
        Ties together all methods to perform the full data preparation stack
        """
        #drop repeated columns
        self.df = self.df.drop_duplicates()
        
        self.drop_columns(columns=drop_columns)

        #prepare datetimes
        self.prepare_datetimes(add_decade=add_decade)

        #encode the categoricals to integer
        if cat_columns is not None:
            self.encode_categoricals(columns=cat_columns)

        #encode datetime (earliest = 0, latest = N days from earliest)
        self.encode_datetime()

        #encode the genres via one-hot
        self.encode_genre(topk=topk_genres)

        #encode album artists (one-hot style)
        self.encode_album_artists()

        #set all dtypes
        self.set_dtypes()

        for column in text_columns:
            if column in self.df.columns:
                self.encode_text(column,n_components=n_components_text)

        #drop missing
        self.df = self.df.dropna()

        self.df = self.df.reset_index(drop=True)