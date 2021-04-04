import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset,DataLoader
import os
import sys
import time
import pickle
import numpy as np
import math
import random
import matplotlib.pyplot as plt


SEQ_LENGTH = 4
VEC_LENGTH = 30720
BATCH_SIZE = 35
learning_rate = 0.000001
TRAIN = 1200

#获取label
with open("/media/lscsc/export/qiaohui/pytorch/code/status.txt", "r", encoding="utf-8") as f:
        text = f.read()
label_txt = []   
for i in range(len(text)//3):
    k = 3*i
    label_txt.append(text[k])
data_x=[]
data_y=label_txt


#加载图片对应特征向量
for i in range(1,1501):
    if i < 10:
            number = '00000' + str(i)
    if i>=10 and i<=99:
        number = '0000' + str(i)
    if i>=100 and i < 1000:
        number = '000' + str(i)
    if i>=1000:
        number = '00' + str(i)
    for j in range(1,5):
        pkl_file = open('/media/lscsc/export/qiaohui/new/vector/'+number+'/vector_'+str(j)+'.pkl',"rb")
        vector = pickle.load(pkl_file)
        data_x.append(vector)






#获取训练集
def get_train_data(batch_size=4,time_step=4,train_begin=0,train_end=4*TRAIN):
    #batch_index=[]
    data_train=data_x[train_begin:train_end]
    # normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(int(len(data_train)/time_step)):
    #    if i % batch_size==0:
    #        batch_index.append(i)
       x=data_train[4*i:4*i+time_step]
       y=data_y[i]
       train_x.append(x)
       train_y.append(y)
    # print normalized_train_data
    #batch_index.append((len(data_train)-time_step))
    return train_x,train_y





# rnn = nn.LSTM(3*5*2048,10,2)
#把所有array转化成tensor，输出[4,30720]
#形式如下：
#[img1
# img2
# img3
# img4]
def turn_into_tensor(x):
    bag = []
    x_tensor = torch.zeros((4,3*5*2048))
    for i in range(len(x)):
        x_tensor = torch.zeros((4,3*5*2048))
        for j in range(4):
            x_tensor[j] = (torch.Tensor(x[i][j]))
        bag.append(x_tensor)
    return bag

#得到打包好的训练集和验证集
x_train,y_train = get_train_data()
x_train = turn_into_tensor(x_train)
x_test,y_test = get_train_data(train_begin=4*TRAIN,train_end=4*1500)
x_test = turn_into_tensor(x_test)


#将数据打包成dataset
class Mydata(Dataset):
    def __init__(self,whole_x_collection,whole_y_collection):
      self.x = whole_x_collection
      self.label = whole_y_collection


    def __getitem__(self,idx):
       tensor = self.x[idx]
       label = self.label[idx]
       return tensor,label

    def __len__(self):
        return (len(self.x))

train_set = Mydata(x_train,y_train)
test_set = Mydata(x_test,y_test)
train_set = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_set = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)


#网络模型
class Classifier(nn.Module):
    def __init__(self, kwargs):
        super(Classifier, self).__init__()

        self.lstm = nn.LSTM(input_size=kwargs['vector_length'], hidden_size=256, num_layers=2)
        self.fc1 = nn.Linear(4 * 256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, kwargs['output_dim'])
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor, ph, pc):
        feature, (h, c) = self.lstm(input_tensor, (ph, pc))

        feature = feature.permute((1,0,2))
        feature = torch.reshape(feature, (BATCH_SIZE, -1))
        x = self.relu(self.fc1(feature))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        x = self.dropout(x)
        x = self.softmax(x)

        return x

cls = Classifier(
    kwargs={'seq_length':4, 'vector_length':30720, 'output_dim':3}
)

#计算平均数，用以计算平均accuracy和loss
def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)



def train(dataloader):
    model.train()
    loss_bag = []
    acc_bag = []
    count = 0
    for batch, (X, y) in enumerate(dataloader):
        count+=1
        if batch == len(dataloader.dataset)//BATCH_SIZE -1:
                break 
        train_correct = 0
        selected_label=np.array(y).astype(int)
        selected_label=torch.from_numpy(selected_label)
        selected_label = selected_label.cuda()
        selected_tensor=X.cuda()
        x = selected_tensor.permute(1,0,2)
        ph0, pc0 = torch.zeros(size=(2, BATCH_SIZE, 256)).cuda(), torch.zeros(size=(2, BATCH_SIZE, 256)).cuda()
        optimizer.zero_grad()
        output = model(x, ph0, pc0)#一个batch出来output
        # selected_label = torch.autograd.Variable(selected_label.long()).cuda()
        loss = criterion(output, selected_label)
        loss_bag.append(loss/BATCH_SIZE)
        loss.backward()
        optimizer.step()     
        for j in range(len(selected_label)):
                if output.max(1)[1][j] == selected_label[j]:
                    train_correct += 1
        accu = train_correct/BATCH_SIZE
        acc_bag.append(accu)
        if batch % 20 == 0:
            loss, current = loss.item(), batch
            print(f"loss: {loss/BATCH_SIZE:>7f}  [{batch:>5d}/{TRAIN//BATCH_SIZE:>5d}],Accuracy:{(100*(train_correct/BATCH_SIZE)):>0.1f}%")
    avg_loss = averagenum(loss_bag)  
    avg_acc = averagenum(acc_bag)
    print(f"Train loss: {avg_loss:>7f},Train accuracy:{(100*(avg_acc)):>0.1f}%")  
        
    # Add parameters' gradients to their values, multiplied by learning rate
    # for p in model.parameters():
    #     p.data.add_(p.grad.data, alpha=-learning_rate)

    return avg_loss,avg_acc

def test(dataloader):
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        count = 0
        for batch, (X, y) in enumerate(dataloader):
            count+=1 
            if batch == len(dataloader.dataset)//BATCH_SIZE -1:
                break   
            selected_label=np.array(y).astype(int)
            selected_label=torch.from_numpy(selected_label)
            selected_label = selected_label.cuda()
            selected_tensor=X.cuda()
            x = selected_tensor.permute(1,0,2)
            ph0, pc0 = torch.zeros(size=(2, BATCH_SIZE, 256)).cuda(), torch.zeros(size=(2, BATCH_SIZE, 256)).cuda()
            pred = model(x, ph0, pc0)#一个batch出来output
            test_loss += criterion(pred,selected_label).item()
            for i in range(len(selected_label)):
                if pred.max(1)[1][i] == selected_label[i]:
                    correct += 1
            # correct += (pred.argmax(1) == selected_label).type(torch.float).sum().item()
    test_loss /= BATCH_SIZE*count
    correct /= BATCH_SIZE*count
    print(f"Test loss: {test_loss:>8f},Test Accuracy: {(100*correct):>0.1f}% \n")

    return correct


model = cls
device = torch.device('cuda:0')
model.to(device)
criterion = nn.CrossEntropyLoss()
epochs = 300
current_accuracy = 0
filepath = os.path.join('/media/lscsc/export/qiaohui/new', 'checkpoint_model_epoch_{}.pth.tar') #保存最优模型
correct_bag = []
train_correct_bag=[]
train_loss_bag=[]

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40, 50], 0.1, last_epoch=-1)
optimizer.zero_grad()
batch_number_collection = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    avg_loss,avg_acc = train(train_set)
    train_correct_bag.append(avg_acc)
    train_loss_bag.append(avg_loss)
    scheduler.step()


    new_correct = test(test_set)
    correct_bag.append(new_correct)
    if avg_acc > current_accuracy:
        torch.save(model, filepath)
    current_accuracy = avg_acc
    
    #画图，train accuracy，train loss以及test accuracy
    plt.figure()
    plt.plot(range(1,len(train_correct_bag)+1),train_correct_bag)
    plt.title('Train Average Accuracy')
    plt.savefig('/media/lscsc/export/qiaohui/new/Train Average Accuracy.jpg')
    plt.figure()
    plt.plot(range(1,len(train_loss_bag)+1),train_loss_bag)
    plt.title('Train Average Loss')
    plt.savefig('/media/lscsc/export/qiaohui/new/Train Average Loss.jpg')
    plt.figure()
    plt.plot(range(1,len(correct_bag)+1),correct_bag)
    plt.title('Test Accuracy')
    plt.savefig('/media/lscsc/export/qiaohui/new/Test Accuracy.jpg')

print("Done!")

