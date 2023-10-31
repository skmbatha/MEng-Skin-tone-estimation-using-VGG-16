"""
Control software version 3

New features

1. Continue from previous session (save all required variables in the program but
   skip creation of folders and fildes).
2. Added : skip mode. In case the system was interrupted and you want to fast-forward to a 
   specific mode, it's doable.
3. Reduced waiting time between "start capture" button press to "actual start".

"""


import sys
import os
import time
import json
import numpy as np
from uart import UART
from ppadb.client import Client as AdbClient


# Main class
class MAIN:
	
	def __init__(self):
		# read saved config from memory
		try:
			
			f=open("light_config.json","r")
			data=f.read()
			f.close()
			
			#Load the data
			data=json.loads(data)
			self.lights=data
		except:
			self.lights={
			"lines":[1,4,5,6,9,10,11,13,17,19,20,21,22,16,27],#GPIO
			"index":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]#LIGHT NUMBER
			}
			
		# reset lights
		self.reset_lights()
		
		# modes
		self.modes=[
		'front-facing',
		'left-facing',
		'right-facing',
		'up-facing',
		'down-facing'
		]
		
		# spectral readings
		print("Read spectrometer? 1:yes, 0:no.")
		self.flag_read_spectrum=self.input()
		
		# init adb device
		try:
			client = AdbClient(host="127.0.0.1", port=5037)
			self.a32=client.devices()[0]
		except:
			print("ADB connection failed, continue? 1:yes, 0:no")
			if self.input()!=1:
				sys.exit(1)
				
		# heat up halo
		print("Heat up halogen lights? 1:yes, 0:no")
		if self.input()==1:
			print("For how long:",end='')
			self.heat_up_halo(self.input())
		
				
		# enter participant's id
		print("Enter participant's id: ",end="")
		self.id=self.input()
		
		print("New session? 1:yes, 0:no")
		if self.input()==1:
			# Create folders for a new participant
			try:
				os.mkdir("data/{}".format(str(self.id)))
				for mode in self.modes:
					os.mkdir("data/{}/{}".format(str(self.id),mode))
			except:
				os.system("rm -R data/{}".format(str(self.id)))
				os.mkdir("data/{}".format(str(self.id)))
				for mode in self.modes:
					os.mkdir("data/{}/{}".format(str(self.id),mode))
					
			# Participant's monk scale value
			print("Enter participant's monk scale value: ",end="")
			self.monk_scale_value=self.input()
			f=open("data/{}/monk_scale_value.json".format(str(self.id)),"w")
			f.write(json.dumps({
			"value":self.monk_scale_value
			}))
			f.close()
			
	def heat_up_halo(self,on_time):
		print("Heating up halogen lights for {} seconds.".format(str(on_time)))
		self.light_on(13)
		self.light_on(14)
		self.light_on(15)
		time.sleep(int(on_time))
		self.light_off(13)
		self.light_off(14)
		self.light_off(15)

	def reset_lights(self):
		print("Resetting all GPIOs to zero.")
		for gpio in self.lights["lines"]:
			os.system("echo \"{}\" > /sys/class/gpio/export".format(gpio))
			os.system("echo \"out\" > /sys/class/gpio/gpio{}/direction".format(gpio))
			os.system("echo \"0\" > /sys/class/gpio/gpio{}/value".format(gpio))
		   
		time.sleep(1)
		
	def light_on(self,value):
		try:
			index=self.lights["index"].index(value)
		except:
			print("Light on: Failed to find light index")
			return
		gpio=self.lights["lines"][index]
		os.system("echo \"1\" > /sys/class/gpio/gpio{}/value".format(str(gpio)))
			
	def light_off(self,value):
		try:
			index=self.lights["index"].index(value)
		except:
			print("Light off: Failed to find light index")
			return
		gpio=self.lights["lines"][index]
		os.system("echo \"0\" > /sys/class/gpio/gpio{}/value".format(str(gpio)))
			
	def calibrate(self):
		
		for i in range(0,15):
			os.system("echo \"1\" > /sys/class/gpio/gpio{}/value".format(str(self.lights["lines"][i])))
			print(">",end="")
			light_number=int(input())
			os.system("echo \"0\" > /sys/class/gpio/gpio{}/value".format(str(self.lights["lines"][i])))
			self.lights["index"][i]=light_number
			print("GPIO {} -> light {}".format(self.lights["lines"][i],light_number))
		
		f=open("light_config.json","w")
		f.write(json.dumps(self.lights))
		f.close()
		
	def input(self):
		try:
			return int(input())
		except:
			return ""


# Main entry point
if __name__ == "__main__":
	
	uart=UART()
	main=MAIN()
	#while True:
	#	a=main.input()
	#	main.reset_lights()
	#	main.light_on(a)
		
		 
	#modes loop
	i=0
	while i<len(main.modes):
		
		spectral_data=[]
		print("\nMode: "+main.modes[i])

		print("Skip mode? 1:yes, 0:no")
		if main.input()==1:
			i+=1
			continue

		print("Press ENTER to continue...")
		main.input()
		print("Starting...")
		time.sleep(1)
		
		# Loop over lights
		for light_number in range(1,16):
			# light on
			if   (light_number==1):
				main.light_on(1)
				main.light_on(2)
			elif (light_number==2):
				main.light_on(3)
			elif (light_number==3):
				main.light_on(2)
				main.light_on(3)
			elif (light_number==4):
				main.light_on(4)
				main.light_on(5)
			elif (light_number==5):
				main.light_on(6)
			elif (light_number==6):
				main.light_on(5)
				main.light_on(6)
			elif (light_number==11):
				main.light_on(10)
				main.light_on(11)
			elif (light_number==13):
				main.light_on(13)
			elif (light_number==14):
				main.light_on(14)
			elif (light_number==15):
				main.light_on(14)
				main.light_on(15)
			else:
				main.light_on(light_number)
				
				
			# delay
			if   (light_number>=13):
				time.sleep(2)
			elif(light_number==1):
				time.sleep(2)
			else:
				time.sleep(1)
				
			
			# Capture
			try:
				# Capture and error checking
				while True:
					
					
					#=================================================================
					try:
						before=main.a32.shell("ls /sdcard/DCIM/Camera/ | wc -l")
						main.a32.shell("input tap 544 2044")
						
						time.sleep(1)
					
						# Check if the image captured
						after=main.a32.shell("ls /sdcard/DCIM/Camera/ | wc -l")
						print("Before:{}; After:{}".format(before,after))
						if after==before:
							print("Trying again...")
							time.sleep(4)
							main.a32.shell("input tap 766 2066")
						else:
							time.sleep(0)
							break
						
					except:
						print("Failed to capture, auto-fixing and continuing...")
						time.sleep(1)
						
						try:
							main.a32.shell("input tap 766 2066")
						except:
							pass
							
						continue
					#=================================================================
						
			except:
				try:
					print("Failed to capture, fix connection and retry, press ENTER")
					main.input()
					#i-=1
					
					print("Tap at ok? 1:yes,0:no")
					if main.input()==1:
						main.a32.shell("input tap 766 2066")
					
					'''time.sleep(1)
					main.a32.shell("input tap 766 2066")
					
					#capture and error checking
					before=main.a32.shell("ls /sdcard/DCIM/Camera/ | wc -l")
					main.a32.shell("input tap 544 2044")
					time.sleep(1)
					after=main.a32.shell("ls /sdcard/DCIM/Camera/ | wc -l")
					print("Before:{}; After:{}".format(before,after))
					if int(after)==int(before):
						time.sleep(1)
						main.a32.shell("input tap 766 2066")
					else:
						break'''
							
					
				except:
					print("Failed to capture, you must restart mode!")
			
			# read spectrum
			if main.flag_read_spectrum==1:
				while True:
					try:
						uart.write("ATDATA")
						reading=uart.get_average_reading(1)
						spectral_data.append(reading)
						break
					except:
						print("Failed to read spectrometer value, retrying... ")
						time.sleep(0)
						pass
						
			#time.sleep(1.5)
			# light off
			
			if   (light_number==1):
				main.light_off(1)
				main.light_off(2)
			elif (light_number==2):
				main.light_off(3)
			elif (light_number==3):
				main.light_off(2)
				main.light_off(3)
			elif (light_number==4):
				main.light_off(4)
				main.light_off(5)
			elif (light_number==5):
				main.light_off(6)
			elif (light_number==6):
				main.light_off(5)
				main.light_off(6)
			elif (light_number==11):
				main.light_off(10)
				main.light_off(11)
			elif (light_number==13):
				main.light_off(13)
			elif (light_number==14):
				main.light_off(14)
			elif (light_number==15):
				main.light_off(14)
				main.light_off(15)
			else:
				main.light_off(light_number)
			
			
		# Repeat?
		print("Repeat mode? 1:yes, 0:no.")
		if main.input()==1:
			main.a32.shell("rm -R /sdcard/DCIM/Camera/*")
			continue
		
		# Save all captured images in "data/self.id/modes[i]"
		os.system("adb pull /sdcard/DCIM/Camera/. data/{}/{}".format(main.id,main.modes[i]))
		
		# Clear directory in android device
		main.a32.shell("rm -R /sdcard/DCIM/Camera/*")
		
		# Save spectral data in "data/self.id/modes[i]/"
		if main.flag_read_spectrum==1: 
			data=json.dumps(spectral_data)
			f=open("data/{}/{}/{}.json".format(main.id,main.modes[i],main.id),"w")
			f.write(data)
			f.close()
		
		# Next modes
		i+=1
	
	
	
