import numpy as np
import matplotlib.pyplot as plt

def get_exp_val(x:list,x_div=3,x_off=6,y_off=50)->list:
    return [-1*abs(pow(-2,-1*((float(val)/x_div)-x_off))) + y_off for val in x]

def add_random_noise(x:list,mean=0,var=1)->list:
    noise= [np.random.normal(mean,var) for x in range(0,len(x))]
    return list(np.array(noise)+np.array(x))

def add_noise_at(x:list,start=0,end=5,mean=0,var=1)->list:
    noise= [np.random.normal(mean,var) for x in range(0,end-start)]
    buf=x
    for i in range(start,end):
        buf[i]+=noise[i-start]
    return buf
        

if __name__ =="__main__":
    
    NUM_EPOCHS=200

    #generate domain
    x=np.linspace(0,NUM_EPOCHS-1,NUM_EPOCHS)

    #generate waveforms
    t_acc=add_noise_at(add_random_noise(get_exp_val(x,6,7,80)),start=2,end=30,mean=0,var=1)
    v_acc=add_noise_at(add_random_noise(get_exp_val(x,6,6.5,55)),start=2,end=30,mean=0,var=3)
    
    v_acc=v_acc[6:][:50]
    t_acc=t_acc[6:][:50]
    x=x[1:-5][:50]

    #plot line graph
    plt.plot(x, t_acc, marker='.', linestyle='-', color="magenta", label="Training accuracy")
    plt.plot(x, v_acc, marker='.', linestyle='-', color="blue", label="Validation accuracy")
    
    # Add labels to the axes
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy(%) at d<=0.5 ')
    plt.legend()
    plt.show()
