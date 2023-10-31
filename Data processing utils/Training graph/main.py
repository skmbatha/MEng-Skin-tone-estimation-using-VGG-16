import os
import numpy as np
import matplotlib.pyplot as plt

_save_=True
BASE_DIR="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\CNN_training_models\\VGG16-PyTorch_Regression - Lab\\Experiment\\pretrained (number of layers freezed )"

_play_=False
LOGS = "C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\CNN_training_models\\VGG16-PyTorch_Regression - Lab\\Experiment\\pretrained (LR)\\1  - 10-29- 08h46m\\training_logs.txt"

if __name__ =="__main__":
   dirs= os.listdir(BASE_DIR)
   for dir in dirs:
        
        f = open(f"{BASE_DIR}\\{dir}\\training_logs.txt")
        lines = f.readlines()
        f.close()

        # Generate acc plots
        v_acc = []
        t_acc = []
        loss  = []
        for i in range(0,len(lines)):
            try:
                if lines[i][1] == '*':
                    v_acc.append(float(lines[i].split(' ')[3]))
                    loss.append(float(lines[i-1].split('\t')[2].split(' ')[1]))
            except:
                pass

        for i in range(1,len(lines)):
            if lines[i].split(' ')[0] == 'Test:' and lines[i-1].split(' ')[0] == 'Epoch:':
                t_acc.append(float(lines[i-1].split('\t')[4].split(' ')[2][1:-1]))

        #generate domain
        x=np.linspace(0,len(v_acc)-1,len(v_acc))

        #enforce same size
        m=min([len(x),len(t_acc),len(v_acc),len(loss)])

        #plot line graph
        plt.plot(x[0:m], t_acc[0:m], marker='.', linestyle='-', color="magenta", label="Training accuracy")
        plt.plot(x[0:m], v_acc[0:m], marker='.', linestyle='-', color="blue", label="Validation accuracy")
        
        # Add labels to the axes
        plt.xlabel('Number of epochs')
        plt.ylabel('Accuracy(%) at d<=0.5 ')
        plt.legend()

        if _save_:
            plt.savefig(f"{BASE_DIR}\\{dir}\\training_graph.jpg")
            plt.clf()
        elif _play_:
            plt.pause(5)
            plt.clf()
        elif _save_!=True and _play_!=True:
            plt.show()
            break
