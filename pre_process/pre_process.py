import os
import glob

datapath="output"
outputpath="result"
MIN_POINTS=20

frames=os.listdir(datapath)
for i in range(len(frames)):
	if i%10==0:
		inputpath=os.path.join(datapath,frames[i])
		list_pc=glob.glob(inputpath + '/*.txt'); list_pc.sort()

		for pc_file in list_pc:
			para_file=pc_file.replace("txt","para")
			pc_f=open(pc_file,"r")
			pcs=len(pc_f.readlines())
			para_f=open(para_file,"r")
			para=para_f.read().split(" ")
			label,x,y,z,l,w,h,rx,ry,rz=para
			pc_f.close()
			para_f.close()

			label="Bus" if label=="Largeandmediumsizedpassengercars" else label
			# print(para_file,pcs,label,x,y,z,l,w,h,rx,ry,rz)
			print("processing...",pc_file,pcs,label,x,y,z,l,w,h,rx,ry,rz)
			if label in ["Minibus"] and pcs>MIN_POINTS:
				output_pc_file=os.path.join(outputpath,label+os.path.basename(pc_file))
				output_para_file=output_pc_file.replace("txt","para")
				os.system("cp {} {}".format(pc_file,output_pc_file))
				with open(output_para_file,"w") as para_out:
					para_out.write("{} {} {} {} {} {} {}".format(x,y,z,l,w,h,rz))
				print("done...")


