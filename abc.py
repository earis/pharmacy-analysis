
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
from skopt.space import Integer, Real
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, brier_score_loss, roc_curve,f1_score, average_precision_score
from skopt import gp_minimize


# transfer learning0: the original model and transfer learning: the target model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = pd.read_csv("C:/pytorch_m/R/data_6_1.csv",encoding="utf-8")
data3 = pd.read_csv("C:/pytorch_m/R/data_6_3.csv",encoding="utf-8")
X_0 = data[["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10"]]
X_3 = data3[["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10"]]
Y_0 = data["event1"] 
Y_0 = data["survival_times1"] * 1e-5
Y_03 = data3['survival_times1'] * 1e-5
Y_13 = data3['survival_times2'] * 1e-5
Y_1 = data["event2"] 
Y_1 = data["survival_times2"]*1e-5


X1_train, X1_test, y1_train, y1_test = train_test_split(X_0, Y_1, test_size=0.2, random_state=123)

# 确保数据是numpy数组
X1_train = X1_train.values if isinstance(X1_train, pd.DataFrame) else X1_train
X1_test = X1_test.values if isinstance(X1_test, pd.DataFrame) else X1_test
y1_train = y1_train.values if isinstance(y1_train, pd.Series) else y1_train
y1_test = y1_test.values if isinstance(y1_test, pd.Series) else y1_test
X0_train, X0_test, y0_train, y0_test = train_test_split(X_0, Y_0, test_size=0.2, random_state=123)

# 确保数据是numpy数组
X0_train = X0_train.values if isinstance(X0_train, pd.DataFrame) else X0_train
X0_test = X0_test.values if isinstance(X0_test, pd.DataFrame) else X0_test
y0_train = y0_train.values if isinstance(y0_train, pd.Series) else y0_train
y0_test = y0_test.values if isinstance(y0_test, pd.Series) else y0_test

X00_train, X00_test, y00_train, y00_test = train_test_split(X_3, Y_03, test_size=0.2, random_state=123)
X00_train = X00_train.values if isinstance(X00_train, pd.DataFrame) else X00_train
X00_test = X00_test.values if isinstance(X00_test, pd.DataFrame) else X00_test
y00_train = y00_train.values if isinstance(y00_train, pd.Series) else y00_train
y00_test = y00_test.values if isinstance(y00_test, pd.Series) else y00_test

Good_layers = 0

class my_linear(nn.Module):
    def __init__(self):
        super(my_linear, self).__init__()
        self.fc11 = nn.Linear(10, 20)
        self.fc12 = nn.Linear(20, 40)
        self.fc13 = nn.Linear(40, 80)
        self.fc14 = nn.Linear(80, 160)
        self.fc15 = nn.Linear(160, 320)
        self.fc16 = nn.Linear(320, 640)
        self.fc17 = nn.Linear(640, 320)
        self.fc18 = nn.Linear(320, 160)
        self.fc19 = nn.Linear(160, 80)
        self.fc110 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20, 10)
        self.fc5 = nn.Linear(10, 5)
        self.fc6 = nn.Linear(5, 1)
    def forward(self, x):
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc13(x))
        x = F.relu(self.fc14(x))
        x = F.relu(self.fc15(x))
        x = F.relu(self.fc16(x))
        x = F.relu(self.fc17(x))
        x = F.relu(self.fc18(x))
        x = F.relu(self.fc19(x))
        x = F.relu(self.fc110(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x)) 
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def train_original_model(params):
    '''Train the original model on the original dataset'''
    #train
    layers,learning_rate = params
    input_dim = X1_train.shape[1]
    # model = linear(10,layers)
    model = my_linear()
    model = model.to(device)
    # loss = nn.CrossEntropyLoss()
    loss = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 500
    for epoch in range(num_epochs):
        
        model.train()
        running_loss = 0.0
        print('Epoch:',epoch)
        # for i in range(0, len(X1_train), batch):
            # X_batch = X1_train[i:i + batch]
            # y_batch = y1_train[i:i + batch]
        X_batch = torch.tensor(X1_train, dtype=torch.float32).to(device)
        y_batch = torch.tensor(y1_train, dtype=torch.float32).to(device)
        y_train_pred = model(X_batch)
        # y_train_pred[y_train_pred < 0.5] = 0
        # y_train_pred[y_train_pred >= 0.5] = 1   
        y_train_pred = y_train_pred.squeeze()
        # y_train_pred = torch.tensor([0 if i < 0.5 else 1 for i in y_train_pred],d).squeeze()
        loss_train = loss(y_train_pred, y_batch)
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        running_loss+=loss_train.item() 
        if running_loss < 1500:
            break
        # print(str(loss_train.item()))
        print(f'Epoch {epoch}, Loss {running_loss}')
        
def transfer_learning(params,params_origin):
    '''Transfer learning from the original model to the new dataset'''
    batch2 = 40000
    layers,learning_rate = params_origin
    modified_hidden_layers, learning_rate = params
    num_epochs = 500
    #load model
    model_ft = my_linear()
    # model_ft = linear(input_dim, layers)    
    model_ft.load_state_dict(torch.load('my_original_model_0.pth'))
    for param in model_ft.parameters():
        param.requires_grad = False
    if modified_hidden_layers == 1:
        num_ftrs = model_ft.fc6.in_features
        model_ft.fc6 = nn.Linear(num_ftrs, 1)
    elif modified_hidden_layers == 2:
        num_ftrs = model_ft.fc5.in_features
        model_ft.fc5 = nn.Linear(num_ftrs, 5)
        model_ft.fc6 = nn.Linear(5, 1)
    else:
        num_ftrs = model_ft.fc_end_be.in_features
        for i in range(modified_hidden_layers - 2):
            setattr(model_ft, f'fc{i}', nn.Linear(num_ftrs, num_ftrs))
        setattr(model_ft, 'fc_end_be', nn.Linear(num_ftrs, 5))
        setattr(model_ft, 'fc_end', nn.Linear(5, 1))
    model_ft = model_ft.to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    for epoch in range(num_epochs):
        model_ft.train()
        running_loss = 0.0
        # for i in range(0, len(X0_train[:40000]), batch2):
        #     X_batch = X0_train[i:i + batch2]
        #     y_batch = y0_train[i:i + batch2]
        X_batch =torch.tensor(X00_train, dtype=torch.float32).to(device)
        y_batch = torch.tensor(y00_train, dtype=torch.float32).to(device)
        y_batch_pred = model_ft(X_batch)
        y_batch_pred = y_batch_pred.squeeze()
        loss = criterion(y_batch_pred, y_batch)
        optimizer_ft.zero_grad()
        loss.backward()
        optimizer_ft.step()
        running_loss += loss.item()
        print(f'Epoch {epoch}, Loss {running_loss}')
    torch.save(model_ft.state_dict(), 'C:/pytorch_m/R/my_transfer_model_0.pth')
    model_ft.eval()
    X_batch =torch.tensor(X00_test, dtype=torch.float32).to(device)
    y_batch = torch.tensor(y00_test, dtype=torch.float32).to(device)
    y_batch_pred = model_ft(X_batch)
    y_batch_pred = y_batch_pred.squeeze()
    loss = criterion(y_batch_pred, y_batch)
    print(f'Final Loss: {loss.item()}')

if __name__ == '__main__':
    params = [14, 0.01]
    train_original_model(params)
    res_transfer = [2, 0.01]
    transfer_learning(res_transfer,params)