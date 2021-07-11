from trainer import BertTrainer
import data_generator
from models import LikenessNetworkSingle, LikenessNetworkMultiple
import env_data
from torch.utils.data import TensorDataset, Dataset, DataLoader

class BertDataset(Dataset):

    def create_dataset(self, device):
        embedder = data_generator.BertClassifier(device)
        x_p, y_p = data_generator.generate_positive_sample(embedder, device=device)
        x_n, y_n = data_generator.generate_negative_sample(embedder, 10, device=device)

        del embedder
        return list(zip(x_p, y_p)) + list(zip(x_n, y_n))

    def __init__(self, device):
        self.x, self.y = zip(*self.create_dataset(device))

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)





dl = DataLoader(BertDataset('cpu'), batch_size=64,pin_memory=True,shuffle=True)

model = LikenessNetworkMultiple().to('cuda:5')
trainer = BertTrainer(model)

losses = trainer.train(dl, 100)







