import torch
import numpy as np
import os
from torchvision import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.models import FaceExpModel
from utils import *
from torchvision.datasets import ImageFolder
from data.dataset import FaceExpressionDataset

class Trainer:
    def __init__(self, trainloader: DataLoader, testloader: DataLoader, output_dir,
                  LR :float=0.01, batch_size: int=8, epochs :int=15):
        
        self.trainloader = trainloader
        self.testloader = testloader
        self.output_dir = create_new_output_dir(output_dir)
        self.lr = LR
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    def create_model(self):
        model = FaceExpModel()

        return model.to(self.device)


    def train(self):
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []

        min_test_loss = np.Inf
        model = self.create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr = self.lr)

        for i in range(self.epochs):
            train_loss, train_acc = self.train_func(model, optimizer, i)
            test_loss, test_acc = self.eval_func(model, i)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_loss < min_test_loss :
                self.save_model(model)
                print("saved-best-weight")
                min_test_loss = test_loss
            model_path = os.path.join(self.output_dir,'last_weight.pt')
            torch.save(model.state_dict(), model_path)
            print('save-last-weight')
            view_visualize(train_losses,test_losses,train_accs,test_accs,self.output_dir)            


    
    def train_func(self, model, optimizer, current_epoch):
        model.train()
        total_loss = 0.0
        total_acc = 0.0

        tk = tqdm(self.trainloader, desc="EPOCH"+"[TRAIN]"+ str(current_epoch+1)+"/"+str(self.epochs))
        for t, data in enumerate(tk):
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            logits, loss = model(images, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += multiclass_accuracy(logits, labels)
            tk.set_postfix({'loss' : '%6f' %float(total_loss / (t+1)), 'acc' : '%6f' %float(total_acc / (t+1)),})

        return total_loss/len(self.trainloader), total_acc/len(self.trainloader)



    def eval_func(self,model, current_epoch):

        model.eval()
        total_loss = 0.0
        total_acc = 0.0

        tk = tqdm(self.testloader, desc="EPOCH"+"[TEST]"+ str(current_epoch+1)+"/"+str(self.epochs))
        for t, data in enumerate(tk):

            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits, loss = model(images, labels)

            total_loss += loss.item()
            total_acc += multiclass_accuracy(logits, labels)
            tk.set_postfix({'loss' : '%6f' %float(total_loss / (t+1)), 'acc' : '%6f' %float(total_acc / (t+1)),})

        return total_loss/len(self.testloader), total_acc/len(self.testloader)
    def save_model(self, model):
        model_path = os.path.join(self.output_dir,'best_weight.pt')
        torch.save(model.state_dict(), model_path)

def main():
    batch_size = 16
    num_workers = 4
    epochs=50
    # dataset = FaceExpressionDataset(r"data\fer2013.csv")

    # train_folder_path, test_folder_path = dataset.pre_data()

    train_folder_path, test_folder_path = r"data\train", r"data\test"

    train_augs, test_augs = get_transforms()
    trainloader,testloader = create_dataloader(train_folder_path,test_folder_path, train_augs, test_augs, batch_size,num_workers,epochs=50)
    output = r"checkpoints"

    trainer =Trainer(trainloader=trainloader,testloader=testloader,output_dir=output)
    trainer.train()
if __name__ == "__main__":
    main()

