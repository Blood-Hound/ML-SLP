import csv
import math
import matplotlib.pyplot as plt
from random import shuffle
theta1=0.1
theta2=0.2
theta3=0.3
theta4=0.4
bias=0.5
dtheta1=dtheta2=dtheta3=dtheta4=dbias=0

alpha = 0.1
def result(x1,x2,x3,x4):
    return(x1*theta1 + x2*theta2 + x3*theta3 + x4*theta4 + bias)

def sigmoid(res):
    return(1/(1+math.exp(-res)))

def prediction(act):
    return (0 if(act<0.5) else 1)

def error(t1,act):
    return (math.pow(t1-act,2))

def update_dtheta_dbias(x1,x2,x3,x4,t1,act):
    global dtheta1,dtheta2,dtheta3,dtheta4,dbias
    dtheta1 = 2*x1*(t1-act)*(1-act)*act
    dtheta2 = 2*x2*(t1-act)*(1-act)*act
    dtheta3 = 2*x3*(t1-act)*(1-act)*act
    dtheta4 = 2*x4*(t1-act)*(1-act)*act
    dbias   = 2*(t1-act)*(1-act)*act

def update_theta_bias():
    global theta1,theta2,theta3,theta4,bias,dtheta1,dtheta2,dtheta3,dtheta4,dbias,alpha
    theta1 +=  alpha * dtheta1
    theta2 +=  alpha * dtheta2
    theta3 +=  alpha * dtheta3
    theta4 +=  alpha * dtheta4
    bias   +=  alpha * dbias

def split_kfold(data,k):
    return([[data[i+(j*k)] for i in range(len(data)//k)] for j in range(k)])

def validate_data(datas):
    predict_true=0
    err = 0
    for data in datas:
        x1=float(data['x1'])
        x2=float(data['x2'])
        x3=float(data['x3'])
        x4=float(data['x4'])
        t1=float(data['t1'])
        res = float(result(x1,x2,x3,x4))
        act = sigmoid(res)
        predict=prediction(act)
        predict_true += (1 if(float(predict)==float(t1)) else 0)
        err += error(t1,act)
    return(predict_true/20,err/20)

def train_data(datas):
    predict_true=0
    err = 0
    for data in datas:
        x1=float(data['x1'])
        x2=float(data['x2'])
        x3=float(data['x3'])
        x4=float(data['x4'])
        t1=float(data['t1'])
        res = float(result(x1,x2,x3,x4))
        act = sigmoid(res)
        predict=prediction(act)
        predict_true += (1 if(float(predict)==float(t1)) else 0)
        err += error(t1,act)
        update_dtheta_dbias(x1,x2,x3,x4,t1,act)
        update_theta_bias()
    return(predict_true/80,err/80)

def plot(error_training, error_test, accuracy_training, accuracy_test): 
    global alpha 
    plt.figure(1)
    plt.plot(error_training, color='blue', linewidth=2, label = 'training')
    plt.plot(error_test, color='red', linewidth=2, label = 'test')
    plt.title('Sum Error: Learning Rate ' + str(alpha))
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    
    plt.figure(2)
    plt.plot(accuracy_training, color='blue', linewidth=2, label = 'training')
    plt.plot(accuracy_test, color='red', linewidth=2, label = 'test')
    plt.title('Accuracy: Learning Rate ' + str(alpha))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.legend()
    plt.show()

csv_file=open('./iris.csv','r')
datas=list(csv.DictReader(csv_file))
shuffle(datas)

k=5
splitted_data=split_kfold(datas,k)



all_acc_train=[]
all_err_train=[]
all_acc_validate=[]
all_err_validate=[]
for epoch in range(300):
    avg_acc_validate=0
    avg_err_validate=0
    avg_acc_train=0
    avg_err_train=0
    for i in range(k):
        for no,data in enumerate(splitted_data):
            if(i==no):
                acc_validate,err_validate = validate_data(data)
                avg_acc_validate+=acc_validate
                avg_err_validate+=err_validate
            else:
                acc_train,err_train = train_data(data)
                avg_acc_train+=float(acc_train)
                avg_err_train+=float(err_train)
    all_acc_train.append(avg_acc_train/5*100)
    all_err_train.append(avg_err_train/5)
    all_acc_validate.append(avg_acc_validate/5*100)
    all_err_validate.append(avg_err_validate/5)

#for i in all_acc_train:
#    print(i)
plot(all_err_train, all_err_validate, all_acc_train, all_acc_validate)



