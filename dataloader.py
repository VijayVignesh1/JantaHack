import torch
from torch.utils.data import Dataset
import pandas as pd
import csv
import os
import cv2
import torch.nn as nn
import torchvision
from sklearn.metrics import f1_score
class DataLoader(Dataset):
    def __init__(self,csv,train_path,train=True):
        self.csv=pd.read_csv(csv)
        self.train_path=train_path
        self.img_size=224
        # print(self.csv['image_names'][:5])
        self.train=train
        if self.train:
            self.labels=self.csv['emergency_or_not']
        self.images=self.csv['image_names']
    def __getitem__(self,index):
        img=cv2.imread(os.path.join(self.train_path,self.images[index]))
        img=cv2.resize(img,(self.img_size,self.img_size))
        tensor_image=torch.FloatTensor(img)
        tensor_image=tensor_image.permute(2,0,1)
        tensor_image/=255.
        if self.train:
            return tensor_image,self.labels[index]
        else:
            return tensor_image,self.images[index]
    def __len__(self):
        return len(self.images)

# print(iter(b).next()[0].shape)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet=torchvision.models.resnet50(pretrained=True)
        self.fc=torch.nn.Linear(1000,2)
    def forward(self,image):
        out=self.resnet(image)  
        out=self.fc(out)
        return out

device="cuda"
a=DataLoader('train.csv','images')
train_loader=torch.utils.data.DataLoader(dataset=a,batch_size=32,num_workers=0,shuffle=True)
model=ResNet()
learning_rate=1e-5
num_epochs=10
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model=model.to(device)
"""
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images.to(device))
        labels=labels.reshape((labels.shape[0]))
        loss = criterion(outputs.to(device), labels.to(device))
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Track the accuracy
        total = labels.size(0)
        predicted = torch.softmax(outputs,dim=1)
        # predicted[:,-1]=0
        _,predicted=torch.max(predicted, 1)
        correct = (predicted == labels.to(device)).sum().item()
        acc_list.append(correct / total)
        f1=f1_score(labels.cpu().numpy(),predicted.cpu().numpy(),average='weighted')
        if (i + 1) % 10 == 0:
            # print(predicted)
            # print(labels)
            # print((predicted == labels.to(device)).shape)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}, F1: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100, 100*f1))
            #print(correct,total)
torch.save({
'epoch': epoch,
'state_dict': model.state_dict(),
'optimizer' : optimizer.state_dict()},
'checkpoint.epoch.resnet.{}.pth.tar'.format(epoch))
# print(outputs)
"""
checkpoint='checkpoint.epoch.resnet.9.pth.tar'
model=ResNet()
model.load_state_dict(torch.load(checkpoint)['state_dict'])
model=model.to("cuda")
a=DataLoader('test.csv','images',False)
test_loader=torch.utils.data.DataLoader(dataset=a,batch_size=16,num_workers=0,shuffle=False)
key1='image_names'
key2="emergency_or_not"
# key2.replace(u"\xad",u'')
results={ key1 : [],
          key2 : [] }

# df = pd.DataFrame (data, columns = ['First Column Name','Second Column Name',...])
for i,(images,image_name) in enumerate(test_loader):
    outputs=model(images.to(device))
    predicted = torch.softmax(outputs,dim=1)
    _,predicted=torch.max(predicted, 1)
    predicted=list(predicted.data.cpu().numpy())
    # print(image_name)
    results['image_names'].extend(image_name)
    results["emergency_or_not"].extend(predicted)
    
df = pd.DataFrame.from_dict(results)
df.to_csv("result1.csv",index=False)
print("Done..")
# print(results)



