import torch 
import pandas as pd
from model_and_helpers import * 

import optuna
from optuna.trial import TrialState

training_batch_size = 128
testing_batch_size = 256
epoch_size = 20

train_raw = pd.read_csv('./train_data.csv',delimiter='|')
test_raw = pd.read_csv('./test_data.csv',delimiter='|')
movie_meta = pd.read_csv('./movie_meta.csv',delimiter='|')

tuning_dic={"bert_dim":{'low':40,'high':43},
           "lr":{'low':1e-3,'high':1e-1}}

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        super().__init__()

        self.labels = [label for label in df['target_rating']]
        self.user_id = [user_id for user_id in df['user_id']]
        self.sequence_movie_ids = [x for x in df['sequence_movie_ids']]
#         self.sequence_ratings = [float_list_of_strint(x) for x in df['sequence_ratings']]
        self.sequence_ratings = [x for x in df['sequence_ratings']]
        self.sex = [x for x in df['sex']]
        self.age_group = [x for x in df['age_group']]
        self.occupation = [x for x in df['occupation']]
        self.target_movie = [x for x in df['target_movie']]
        
        

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])
    
    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return {'other_features':{'user_id':self.user_id[idx],
                                 'sex':self.sex[idx],
                                 'age_group':self.age_group[idx],
                                 'occupation':self.occupation[idx]},
               'transform_features':{'sequence_movie_ids':self.sequence_movie_ids[idx],
                                    'sequence_movie_ratings':self.sequence_ratings[idx],
                                    'target_movie_id':self.target_movie[idx]}}

    def __getitem__(self, idx):

        batch_features = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_features, batch_y

        

def get_train_validate(train,validation_rate=0.1,training_batch_size=128):
    """
    split train into train and validate
    """
    validation_size = round(len(train)*validation_rate)
    training_size = len(train)-validation_size

    train_set, validation_set = torch.utils.data.random_split(train, [training_size, validation_size])
    
    train_itr = torch.utils.data.DataLoader(train_set, 
                                        batch_size=training_batch_size)
    validation_itr = torch.utils.data.DataLoader(validation_set, 
                                        batch_size=training_batch_size)
    return train_itr,validation_itr,[training_size,validation_size]

train = Dataset(train_raw)
test = Dataset(test_raw)

def get_model(trial):
    bert_dim = trial.suggest_int('bert_dim',tuning_dic['bert_dim']['low'],tuning_dic['bert_dim']['high'])
    model = TransformerRecommender(3,2,2,0.4,bert_dim)
    return model

def objective(trial):
    model = get_model(trial)
    #lr = trial.suggest_float("lr", tuning_dic['lr']['low'],tuning_dic['lr']['high'], log=True)
    lr = 0.0015
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')
    mae_loss=nn.L1Loss(reduction='sum')
    
    train_itr,valid_itr,[train_size,valid_size] = get_train_validate(train)
    
    model.train()
    print('start training')
    for epoch in range(epoch_size):
        total_loss = 0
        valid_total_mae = 0
        for i,batch in enumerate(train_itr):
            if i%500==0:
                print(f'training epoch {epoch} batch {i}')
            feature,label = batch
            label = label.unsqueeze(-1)
            label= label.float()
            output = model(feature)

            loss = criterion(output,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            
        model.eval()
        
        with torch.no_grad():
            for batch in valid_itr:
                feature,label = batch
                label = label.unsqueeze(-1)
                label= label.float()
                output = model(feature)
                valid_total_mae+=mae_loss(output,label).item()
        

        total_loss_avg = total_loss/train_size
        mae_avg = valid_total_mae/valid_size
        
        print(f"epoch {epoch}: loss - {total_loss_avg}, mae - {mae_avg}")
        trial.report(mae_avg, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    trial.set_user_attr(key="best_model", value=model)
            
    return mae_avg

def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=2, timeout=600,callbacks=[callback])

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial
best_model=study.user_attrs["best_model"]

mae_loss=nn.L1Loss(reduction='sum')
mae = 0
test_itr = torch.utils.data.DataLoader(test, 
                                       batch_size=256)
with torch.no_grad():
    for batch in test_itr:

        feature,label = batch
        label = label.unsqueeze(-1)
        label= label.float()
        output = best_model(feature)
        mae+=mae_loss(output,label).item()

print(f'mae on test set is {mae/len(test)}')