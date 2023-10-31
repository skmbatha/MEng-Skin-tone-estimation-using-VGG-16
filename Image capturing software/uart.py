import serial
import time
import numpy as np

ser=serial.Serial("/dev/ttyS0",115200,timeout=1)

class UART:
    
    def __init__(self):
        """
        Initialize the serial interface over seria0 on the raspberry pi 4
        """
        self.ser=serial.Serial("/dev/ttyS0",115200,timeout=1)
        
        
    def write(self,data):
        """
        This function takes an AT command and then
        sends it over UART to serial0(14&15) on the raspberry
        pi 4.
        """
        ser.write(bytes("{}\r".format(data),'utf-8'))
        time.sleep(0.2)
        
    def read(self):
        """
        Reads the AT command receved from the RX buffer.
        The response must be an AT response with an OK at the end
        otherwise the read will not terminate it's reading process/it will
        hang.
        """
        data=""
        while True:
            byte=ser.read()
            data+=byte.decode('utf-8')

            
            if "OK" in data:
                ser.flushInput()
                ser.flushOutput()
                return data

    def get_data(self):
        """Get a single sample from the spectrumeter and then return
           a list of float values representing the amplitude of
           spectrum values from min to max (6 channels).
        """
        self.write("ATDATA")
        data=self.read()
        
        #get list data
        data=data.replace("OK","").strip()
        data=data.split(",")
        for i in range(0,len(data)):
            data[i]=int(data[i].strip())
            
        #return int list data
        return data
         
    def get_average_reading(self,num_samples):
        """
        This function reads the spectrometer values using method
        get_data(). It then calculates an average of the data read.
        The average is calculated over the number of samples specified in
        the argument. 
        """
        data=np.array(self.get_data())
        for i in range(1,int(num_samples)+1):
            data=(data+np.array(self.get_data()))*0.5
        return list(data)
            
            
if __name__=="__main__":
    
    uart=UART()
    
    while True:
        print (">",end="")
        input_data=input()
        print (input_data)
        uart.write(input_data)
        print(uart.read())
