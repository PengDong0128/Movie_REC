import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
import itertools
from transformers import DistilBertTokenizer, DistilBertModel

sequence_length = 4

def float_list_of_str(l):
    return [float(x) for x in l.split(',')]
def build_vocab(seq,unknown='<unk>'):
    """
    Building vocabulary
    """
    counter = Counter(seq)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    v1 = vocab(ordered_dict)

    unk_token = unknown
    if unk_token not in v1: 
        v1.insert_token(unk_token, 0)
    v1.set_default_index(v1[unk_token])
    return v1


train_raw = pd.read_csv('./train_data.csv',delimiter='|')
test_raw = pd.read_csv('./test_data.csv',delimiter='|')
movie_meta = pd.read_csv('./movie_meta.csv',delimiter='|')

# pre calculate bert vectors for every sentence
# bert too heavy to put in the structure (running locally), even forward process can break kernel
# so I created a look up table which is literally using bert vectors to create a fixed movie overview embedding

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

corpus = movie_meta['overview'].values
bert_inputs = tokenizer(list(corpus), return_tensors="pt",padding=True,truncation=True,max_length=10)
bert_result_raw = bert_model(**bert_inputs,output_hidden_states=True)

bert_embd = torch.mean(torch.mean(torch.stack(bert_result_raw.hidden_states[-2:]),axis=0),axis=1)

# movie metadata look up table
movie_meta_lookup = {}
bert_id = 0
for e in movie_meta.values:
    movie_meta_lookup[e[0]] = {'overview':e[3],'genre':e[-1],'bert_embd':bert_embd[bert_id]}
    bert_id+=1
movie_meta_lookup['unknown']={'bert_embd':torch.mean(bert_embd,axis=0)}

other_features = ['user_id','sex','age_group','occupation']
transform_features = ['sequence_movie_ids','sequence_movie_ratings','target_movie_id']

# construct feature config look up table
feature_conf = {}
feature_conf['other_features']={}
feature_conf['transform_features']={}

for f in other_features:
    v = build_vocab(train_raw[f])
    emb_dim = round(max(2,len(v)**0.5))
    tmp_dic = {'vocab':v,
              'emb_dim':emb_dim,
              'embedding_layer':nn.Embedding(len(v),emb_dim)}
    feature_conf['other_features'][f] = tmp_dic

for f in transform_features:
    if f == 'sequence_movie_ids':
        seq = train_raw['sequence_movie_ids'].values
        seq = [x.split(',') for x in seq]
        seq = [x for x in itertools.chain(*seq)]
        v = build_vocab(seq)
        emb_dim = round(max(2,len(v)**0.5))
        tmp_dic = {'vocab':v,
                  'emb_dim':emb_dim,
                  'embedding_layer':nn.Embedding(len(v),emb_dim),
                  'emb':True}
    else:
        tmp_dic = {'emb':False}
    feature_conf['transform_features'][f] = tmp_dic

# fetch movie embedding and vocabulary for easy usage
movie_emb = feature_conf['transform_features']['sequence_movie_ids']['embedding_layer']
movie_vocab = feature_conf['transform_features']['sequence_movie_ids']['vocab']

# create genre vector lookup table
genre_lookup = []
genre_dim = 18

for m in movie_vocab.vocab.itos_:
    if m in movie_meta_lookup:
        genre = torch.tensor([float(x) for x in movie_meta_lookup[m]['genre'].split(',')])
    else:
        genre = torch.zeros(genre_dim)
    genre = genre.unsqueeze(0)
    genre_lookup.append(genre)
genre_lookup = torch.concat(genre_lookup,dim=0)

# create bert vector lookup table
bert_lookup = []
bert_dim = 768

for m in movie_vocab.vocab.itos_:
    if m in movie_meta_lookup:
        bert_vec = movie_meta_lookup[m]['bert_embd']
    else:
        bert_vec = movie_meta_lookup['unknown']['bert_embd']
    bert_vec = bert_vec.unsqueeze(0)
    bert_lookup.append(bert_vec)
bert_lookup = torch.concat(bert_lookup,dim=0)

class TransformerRecommender(nn.Module):
    def __init__(self,nhead,d_hid,nlayers,dropout,bert_output_dim):
        super().__init__()
        
        # define genre lookup(embedding) table
        self.genre_emb = nn.Embedding(len(movie_vocab),genre_dim)
        self.genre_emb.weight.data.copy_(genre_lookup)
        self.genre_emb.requires_grad_(False)
        
        # define bert lookup table
        self.bert_lookup = nn.Embedding(len(movie_vocab),bert_dim)
        self.bert_lookup.weight.data.copy_(bert_lookup)
        self.bert_lookup.requires_grad_(False)
        
        # fetch movie_id embedding
        self.movie_emb = feature_conf['transform_features']['sequence_movie_ids']['embedding_layer']
        movie_emb_dim = feature_conf['transform_features']['sequence_movie_ids']['emb_dim']
        
        # define  movie_sequence_id_genre_dense (movie_embedding+genre->movie_embedding_dim)
        movie_sequence_input_size = movie_emb_dim+genre_dim
        movie_sequence_output_size = movie_emb_dim
        self.movie_sequence_id_genre_dense = nn.Linear(movie_sequence_input_size,movie_sequence_output_size)
        
        #define transformer layers
        d_model = feature_conf['transform_features']['sequence_movie_ids']['emb_dim']
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        fc_feature_count = d_model*sequence_length+sum([feature_conf['other_features'][x]['emb_dim'] for x in feature_conf['other_features']])+bert_output_dim*sequence_length
        # fully connect layers
        self.l1 = nn.Linear(fc_feature_count,256)
        self.l2 = nn.Linear(256,64)
        self.l3 = nn.Linear(64,1)
        self.bert_l1 = nn.Linear(768,256)
        self.bert_l2 = nn.Linear(256,bert_output_dim)
    def movie_sequence_id_genre_cross(self,m):
        """
        feature crossing for movie_id & genre
        input shape: movie_id_emb_dim+genre_dim
        output shape: movie_id_emb_dim
        """
        genre_ = self.genre_emb(m)
        id_ = self.movie_emb(m)
        concated = torch.concat((id_,genre_),dim=-1)
        Z = self.movie_sequence_id_genre_dense(concated)
        return Z
    
    def forward(self,feature):
        #other features embedding start###
        other_features_tensors = [feature_conf['other_features'][x]['embedding_layer'](
                                  torch.tensor((feature_conf['other_features'][x]['vocab'](feature['other_features'][x]))))
                         for x in other_features]

        other_features_tensors = torch.concat(other_features_tensors,dim=1)
        #other features embedding end###

        #transformer features embedding start###
        ## movie id sequence crossing: (movie_id_embedding,genre_onehot)->linear->(new_embedding+position_encoder)*rating
        movie_sequence_ids = [x.split(',') for x in feature['transform_features']['sequence_movie_ids']]
        movie_sequence_ids = torch.tensor([movie_vocab(x) for x in movie_sequence_ids])

        movie_sequence_ratings = feature['transform_features']['sequence_movie_ratings']
        movie_sequence_ratings = [float_list_of_str(x) for x in movie_sequence_ratings]
        
        movie_id_sequence_emb = self.movie_sequence_id_genre_cross(movie_sequence_ids)

        movie_sequence_ratings_broadcasted = torch.tensor(movie_sequence_ratings).unsqueeze(-1).expand(movie_id_sequence_emb.size())
        encoded_position_broadcasted = torch.arange(sequence_length-1,0,-1).unsqueeze(-1).unsqueeze(0).expand(movie_id_sequence_emb.size())
        sequence_cross = (movie_id_sequence_emb+encoded_position_broadcasted)*movie_sequence_ratings_broadcasted
        ## movie id sequence crossing end
        
        #target movie id embedding start###
        target_movie_id = feature['transform_features']['target_movie_id']
        target_movie_id = torch.tensor(movie_vocab(target_movie_id)).unsqueeze(-1)
        target_movie_emb = self.movie_sequence_id_genre_cross(target_movie_id)
        #target movie id embedding end###
        
        #transformer start###
        transformer_feed_features = torch.concat((sequence_cross,target_movie_emb),dim=1)
        transformed_feature_tensors = self.transformer_encoder(transformer_feed_features).flatten(1)
        transformed_feature_tensors_short_cut = transformed_feature_tensors+transformer_feed_features.flatten(1)
        #transformer end###
        
        #concat embedding features
        embd_features = torch.concat((transformed_feature_tensors,other_features_tensors),dim=1)
        
        #bert block start####
        #movie sequence bert
        movie_sequence_bert = self.bert_lookup(movie_sequence_ids)
        movie_sequence_ratings_broadcasted = torch.tensor(movie_sequence_ratings).unsqueeze(-1).expand(movie_sequence_bert.size())
        encoded_position_broadcasted = torch.arange(sequence_length-1,0,-1).unsqueeze(-1).unsqueeze(0).expand(movie_sequence_bert.size())
        movie_sequence_bert = (movie_sequence_bert+encoded_position_broadcasted)*movie_sequence_ratings_broadcasted
        
        #target movie bert
        target_bert = self.bert_lookup(target_movie_id)
        #concat bert vectors and transform to lower dimension
        bert_vector = torch.concat((movie_sequence_bert,target_bert),dim=1)
        bert_vector = self.bert_l1(bert_vector)
        bert_vector = F.relu(self.bert_l2(bert_vector)).flatten(1)
        #bert block ends###
    
        
        fc_vector = torch.concat((embd_features,bert_vector),dim=1)
        
        l_output = F.leaky_relu(self.l1(fc_vector))
        l_output = F.leaky_relu(self.l2(l_output))
        l_output = F.leaky_relu(self.l3(l_output))
        return l_output
