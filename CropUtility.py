import PIL, glob, os, cv2, math
#import cv2 as cv
import tkinter as tk
import numpy as np

from tkinter import filedialog
from skimage import io, transform,  util
from math import sqrt,cos,sin,radians
from PIL import ImageTk

def clamp(v):
    if v < 0:
        return 0
    if v > 255:
        return 255
    return int(v + 0.5)

class Matrix(object):
    def __init__(self):
        self.matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

    def set_hue_rotation(self, degrees):
        cosA = cos(radians(degrees))
        sinA = sin(radians(degrees))
        self.matrix[0][0] = cosA + (1.0 - cosA) / 3.0
        self.matrix[0][1] = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA
        self.matrix[0][2] = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA
        self.matrix[1][0] = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA
        self.matrix[1][1] = cosA + 1./3.*(1.0 - cosA)
        self.matrix[1][2] = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA
        self.matrix[2][0] = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA
        self.matrix[2][1] = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA
        self.matrix[2][2] = cosA + 1./3. * (1.0 - cosA)
    
    def hue_rotate(self, degrees=1):
        temp = Matrix()
        temp.set_hue_rotation(degrees)
        self.matrix = np.matmul(np.array(self.matrix),temp.toMatrix())
    def stepX(self, a = 1):
        x =a*0.01
        self.matrix[0][3] = self.matrix[0][0]*x
    def stepY(self, a = 1):
        x = 1.00 + a*0.01
        self.matrix[1][3] = self.matrix[1][1]*x
    def stepZ(self, a = 1):
        x = 1.00 + a*0.01
        self.matrix[2][3] = self.matrix[2][2]*x
    def stretchX(self, a = 1):
        x = 1.00 + a*0.01
        self.matrix[0][0] = self.matrix[0][0]*x
    def stretchY(self, a = 1):
        x = 1.00 + a*0.01
        self.matrix[1][1] = self.matrix[1][1]*x
    def stretchZ(self, a = 1):
        x = 1.00 + a*0.01
        self.matrix[2][2] = self.matrix[2][2]*x
    
    def apply_RGB(self, img):
        img_reshaped = img.reshape((img.shape[0]*img.shape[1],img.shape[2]))
        img_reshaped = np.hstack((img_reshaped,np.ones((img.shape[0]*img.shape[1],1))))
        corrected_img = np.delete(np.dot(np.array(self.matrix),img_reshaped.T),3,0)
        corrected_img = np.clip(corrected_img.T.reshape(img.shape),0,255).astype('uint8')
        return corrected_img
    
    def apply_HSV(self, img):
        img_reshaped = img.reshape((img.shape[0]*img.shape[1],img.shape[2]))
        img_reshaped = np.hstack((img_reshaped,np.ones((img.shape[0]*img.shape[1],1))))
        corrected_img = np.delete(np.dot(np.array(self.matrix),img_reshaped.T),3,0)
        corrected_img = np.clip(corrected_img.T.reshape(img.shape),0,1)
        return corrected_img
    
    def apply_single(self, r, g, b):
        rx = r * self.matrix[0][0] + g * self.matrix[0][1] + b * self.matrix[0][2]
        gx = r * self.matrix[1][0] + g * self.matrix[1][1] + b * self.matrix[1][2]
        bx = r * self.matrix[2][0] + g * self.matrix[2][1] + b * self.matrix[2][2]
        return clamp(rx), clamp(gx), clamp(bx)
    
    def toMatrix(self):
        return np.array(self.matrix)
    
    def set_matrix(self, M):
        self.matrix = M
        
    def reset_matrix(self):
        self.matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    
def dif(clr1,clr2):
    if clr1.shape[1] == 3:
        temp = np.zeros((clr1.shape[0],1))
        clr1 = np.concatenate([temp,clr1],axis=1)
    if clr2.shape[1] == 3:
        temp = np.zeros((clr2.shape[0],1))
        clr2 = np.concatenate([temp,clr2],axis=1)
    clr1 = [x.astype(float) for x in clr1]
    clr2 = [x.astype(float) for x in clr2]
    d = []
    for a, b in zip(clr1,clr2):
        t =math.hypot(a[1]-b[1],a[2]-b[2],a[3]-b[3])
        d.append(t)
    return d

def difHSV(clr1,clr2):
    if clr1.shape[1] == 3:
        temp = np.zeros((clr1.shape[0],1))
        clr1 = np.concatenate([temp,clr1],axis=1)
    if clr2.shape[1] == 3:
        temp = np.zeros((clr2.shape[0],1))
        clr2 = np.concatenate([temp,clr2],axis=1)
    clr1 = [x.astype(float) for x in clr1]
    clr2 = [x.astype(float) for x in clr2]
    d = []
    for a, b in zip(clr1,clr2):
        R1 = a[2]
        R2 = b[2]
        theta1 = 360*a[1]
        theta2 = 360*b[1]
        t =math.hypot(R1*math.cos(theta1)-R2*math.cos(theta2),R1*math.sin(theta1)-R1*math.sin(theta2),a[3]-b[3])
        d.append(t)
    return d


class App:
    def __init__(self, master):
        global inputdir, outputdir
        global circle_x, circle_y
        circle_x = tk.IntVar()
        circle_y = tk.IntVar()
        size = 500
        inputdir = ''
        outputdir = ''
        frame = tk.Frame(master)
        frame.grid()
        
        self.x_min = 0
        self.x_max = int(size*4/5)
        self.y_min = 0
        self.y_max = size
        self.current_image = tk.IntVar()
        self.mat = Matrix()
        self.white_balance = False
        self.wb = tk.Button(frame,text="White Balance Off",command=self.toggle_wb)
        self.wb.grid(row=0,column=6)
        self.reset = tk.Button(frame,text="reset",command=self.reset)
        self.reset.grid(row=0,column=9)
        self.T_width = tk.Label(frame, height=1, width=20,text="Pixel Width: ")
        self.T_width.grid(row=3,column =1,columnspan=2,sticky=tk.W)
        
        self.rot = False
        self.ar = tk.Button(frame,text="Rotate 90",command=self.toggle_ar)
        self.ar.grid(row=1,column=6)
        
        self.T_out = tk.Text(frame, height=1, width=50)
        self.T_out.grid(row=1,column=0,columnspan = 3)
        self.output_dir = tk.Button(frame, text="Out", command=self.getoutputdir)
        self.output_dir.grid(row=1,column=3)
        self.T_in = tk.Text(frame, height=1, width=50)
        self.T_in.grid(row=0,column=0,columnspan=3)
        self.input_1= tk.Button(frame, text="In", command=self.getinputdir)
        self.input_1.grid(row=0,column=3)
        self.rotate_forward= tk.Button(frame, text="H Rotate +", command=self.rot_plus)
        self.rotate_forward.grid(row=1,column=9,sticky=tk.W)
        self.rotate_back= tk.Button(frame, text="H Rotate -", command=self.rot_minus)
        self.rotate_back.grid(row=1,column=8,stick=tk.E)
        self.save_single= tk.Button(frame, text="Apply", command=self.apply_once)
        self.save_single.grid(row = 3, column = 8,stick=tk.E)
        self.save_all= tk.Button(frame, text="Apply to all", command=self.apply_all)
        self.save_all.grid(row = 3, column = 9,stick=tk.W)
        self.imgs = []
        self.load_imgs = tk.Button(frame, text= "Load",command=self.load)
        self.load_imgs.grid(row=3,column=4)
        self.next = tk.Button(frame, text="Next", command=self.image_next)
        self.next.grid(row=3,column=2,sticky=tk.E)
        self.back = tk.Button(frame, text="Back", command=self.image_back)
        self.back.grid(row=3,column=0,sticky=tk.W)
        self.canvas1 = tk.Canvas(frame,width = (self.x_max-self.x_min), height = (self.y_max-self.y_min))
        self.canvas1.configure(scrollregion=(self.x_min,self.y_min,self.x_max,self.y_max))
        self.canvas1.grid(row=5,column=0,rowspan = 4, columnspan=4,sticky=tk.W+tk.E)
        self.canvas1.create_line(self.x_min,self.y_min,self.x_min,self.y_max,width=5,fill = "green",tags=("linex1"))
        self.canvas1.create_line(self.x_max,self.y_min,self.x_max,self.y_max,width=5,fill = "green",tags=("linex2"))
        self.canvas1.create_line(self.x_min,self.y_min,self.x_max,self.y_min,width=5,fill = "green",tags=("liney1"))
        self.canvas1.create_line(self.x_min,self.y_max,self.x_max,self.y_max,width=5,fill = "green",tags=("liney2"))
        self.canvas1.create_oval(1,1,1,1,width=2,tags="circle")
        self.canvas2 = tk.Canvas(frame,width = (self.x_max-self.x_min), height = (self.y_max-self.y_min))
        self.canvas2.configure(scrollregion=(self.x_min,self.y_min,self.x_max,self.y_max))
        self.canvas2.grid(row=5,column=6,rowspan = 4, columnspan=4)
        self.right = self.canvas2.create_image(self.x_max/2,self.y_max/2,anchor=tk.CENTER,tags="Alter")
        
        self.raw = np.empty((0,0))
        self.left = np.empty((0,0))
        self.crop_x1 = tk.Scale(frame, from_=0, to=50, orient='horizontal',length=(int((self.x_max-self.x_min)/2)),command=self.draw_crop_x1,showvalue=0)
        self.crop_x1.grid(row=4,column=0,columnspan=2,sticky=tk.W+tk.E)
        self.crop_x2 = tk.Scale(frame, from_=50, to=100, orient='horizontal',length=(int((self.x_max-self.x_min)/2)),command=self.draw_crop_x2,showvalue=0)
        self.crop_x2.grid(row=4,column=2,columnspan=1,sticky=tk.W+tk.E)
        self.crop_x2.set(100)
        
        self.crop_y1 = tk.Scale(frame, from_=0, to=50, orient='vertical',length=(int((self.y_max-self.y_min)/2)),command=self.draw_crop_y1,showvalue=0)
        self.crop_y1.grid(row=5,column=3)
        self.crop_y2 = tk.Scale(frame, from_=50, to=100, orient='vertical',length=(int((self.y_max-self.y_min)/2)),command=self.draw_crop_y2,showvalue=0)
        self.crop_y2.grid(row=6,column=3)
        self.crop_y2.set(100)
    
    def reset(self):
        self.mat.set_hue_rotation(0)
        self.crop_y1.set(0)
        self.crop_y2.set(100)
        self.crop_x1.set(0)
        self.crop_x2.set(100)
        if self.white_balance:
            self.toggle_wb()
        if self.rot:
            self.toggle_ar()
        self.load()
        
    def rot_plus(self):
        self.mat.hue_rotate(10)
        self.update_right()
    
    def rot_minus(self):
        self.mat.hue_rotate(-10)
        self.update_right()
        
    def apply_once(self):
        im = self.left
        name = self.imgs[self.current_image.get()]
        processed_image = self.process_image(im)
        self.image_export(processed_image,name)
        self.image_next()
            
    def apply_all(self):
        for im in self.imgs:
            processed_image = self.process_image(io.imread(im))
            self.image_export(processed_image, im)
            
    def image_export(self,img, name):
        global outputdir
        if not os.path.exists(outputdir):
            return
        name = name.replace("\\",'/')
        name = name.split('/')[len(name.split('/'))-1]
        f = os.path.splitext(name)
        fname = f[0] + "-cropped" + f[1]
        fpath = os.path.join(outputdir +"/", fname)
        io.imsave(fpath,img)
    
    def toggle_wb(self):
        if self.white_balance:
            self.white_balance = False
            self.wb['text'] = "White Balance Off"
            self.wb['relief'] = tk.RAISED
            self.canvas1.unbind('<Button-1>')
        else:
            self.white_balance = True
            self.wb['text'] = "White Balance On"
            self.wb['relief'] = tk.SUNKEN
            self.canvas1.bind('<Button-1>', self.click)
        #self.load()
        
    def toggle_ar(self):
        if self.rot:
            self.rot = False
            self.ar['text'] = "Rotate 0"
            self.ar['relief'] = tk.RAISED
        else:
            self.rot = True
            self.ar['text'] = "Rotate 90"
            self.ar['relief'] = tk.SUNKEN
        self.update_right()
        
    def image_next(self):
        if self.current_image.get() < len(self.imgs)-1:
            self.current_image.set(self.current_image.get()+1)
        else:
            self.current_image.set(0)
        self.load()
        
    def image_back(self):
        if self.current_image.get() > 0:
            self.current_image.set(self.current_image.get()-1)
        else:
            self.current_image.set(len(self.imgs)-1)
        self.load()
        
    def draw_crop_x1(self,val):
        coords = self.canvas1.coords('linex1')
        x = int(int(val)/100*(self.x_max-self.x_min))
        self.canvas1.tag_raise('linex1')
        self.canvas1.coords("linex1",x,coords[1],x,coords[3])
        self.update_right()
        
    def draw_crop_x2(self,val):
        coords = self.canvas1.coords('linex2')
        x = int(int(val)/100*(self.x_max-self.x_min))
        self.canvas1.tag_raise('linex2')
        self.canvas1.coords("linex2",x,coords[1],x,coords[3])
        self.update_right()
        
    def draw_crop_y1(self,val):
        coords = self.canvas1.coords('liney1')
        y = int(int(val)/100*(self.y_max-self.y_min))
        self.canvas1.tag_raise('liney1')
        self.canvas1.coords("liney1",coords[0],y, coords[2],y)
        self.update_right()
        
    def draw_crop_y2(self,val):
        coords = self.canvas1.coords('liney2')
        y = int(int(val)/100*(self.y_max-self.y_min))
        self.canvas1.tag_raise('liney2')
        self.canvas1.coords("liney2",coords[0],y, coords[2],y)
        self.update_right()
        
    def getoutputdir(self):
        global outputdir
        outputdir = filedialog.askdirectory(parent=root,initialdir="/",title='Please select the output directory')
        self.T_out.delete(1.0, tk.END)
        self.T_out.insert(tk.END, outputdir)
        
    def getinputdir(self):
        global inputdir
        input_file = filedialog.askopenfilename(parent=root,initialdir='F:\\Photo Booth Pictures\\',title='Please select the input directory').replace('\\','/')
        if input_file:
            inputdir = os.path.dirname(input_file) + "\\"
        self.imgs = [x.replace('\\','/') for x in glob.glob(inputdir + "*.JPG")]
        for i, x in enumerate(self.imgs):
            if x == input_file:
                self.current_image.set(i)
                break
        self.load()
        self.T_in.delete(1.0, tk.END)
        self.T_in.insert(tk.END, inputdir.replace('\\','/'))
        
    def crop_img(self, img):
        y, x = img.shape[:2]
        m_y = (self.y_max-self.y_min)
        m_x = (self.x_max-self.x_min)
        scale_factor = y/m_y
        if x>y:
            scale_factor = x/m_x
        x1 = self.canvas1.coords('linex1')[0]
        x2 = self.canvas1.coords('linex2')[0]
        y1 = self.canvas1.coords('liney1')[1]
        y2 = self.canvas1.coords('liney2')[1]
        if x1 <= (m_x-x/scale_factor)/2:
            x1 = int(0)
        else:
            x1 = int(scale_factor*x1 - (scale_factor*m_x-x)/2)
        if x2 > m_x - (m_x-x/scale_factor)/2:
            x2 = x
        else:
            x2 = int(scale_factor*x2 - (scale_factor*m_x-x)/2)
        if y1 <= (m_y-y/scale_factor)/2:
            y1 = int(0)
        else:
            y1 = int(scale_factor*y1 - (scale_factor*m_y-y)/2)
        if y2 >= m_y-(m_y-y/scale_factor)/2:
            y2 = y
        else:
            y2 = int(scale_factor*y2-(scale_factor*m_y-y)/2)
        self.T_width.configure(text="Pixel Width: " + str(int((self.raw.shape[0]/y)*(x2-x1))))
        return img[y1:y2,x1:x2]
    
    def load(self):
        global inputdir
        im = self.current_image.get()
        if not os.path.exists(inputdir):
            inputdir = os.getcwd() + "\\"
        self.imgs = glob.glob(inputdir + "*.JPG")
        if len(self.imgs) > 0 :
            raw = io.imread(self.imgs[im])
            self.raw = raw
            self.left = (255*(transform.rescale(raw,0.25,multichannel=True,anti_aliasing=True))).astype(np.uint8)
            left_image = self.photo_image(self.left)
            right_image = self.process_image(self.left)
            right_image = self.photo_image(right_image)
            self.canvas1.create_image(self.x_max/2,self.y_max/2,image=left_image,anchor=tk.CENTER,tags="Original")
            self.canvas1.image = left_image
            self.canvas1.tag_lower('Original')
            self.canvas2.itemconfigure(self.right,image=right_image)
            self.canvas2.image = right_image
            
    def rot_90(self,img):
        return np.rot90(img)
        
    def update_right(self):
        if self.left.any():
            image = self.left
            processed_image = self.process_image(image)
            processed_image = self.photo_image(processed_image)
            self.canvas2.itemconfig(self.right, image= processed_image)
            self.canvas2.image= processed_image
            
    def process_image(self, image):
        if self.white_balance:
            image= self.whitepatch_balancing(image)
        image = self.mat.apply_RGB(image)
        image = self.crop_img(image)
        if self.rot:
            image = self.rot_90(image)
        return image
        
    def whitepatch_balancing(self, img):
        y, x = img.shape[:2]
        m_y = (self.y_max-self.y_min)
        m_x = (self.x_max-self.x_min)
        scale_factor = y/m_y
        if x>y:
            scale_factor = x/m_x
        
        coord = self.canvas1.coords('circle')
        r = int(scale_factor*int((abs(coord[2]-coord[0]))/2))-1
        
        if r == -1:
            return img
        
        x_pos = int(coord[0]+r/2)
        y_pos = int(coord[1]+r/2)
        if x_pos <= (m_x-x/scale_factor)/2:
            x_pos = r
        elif x_pos > m_x - (m_x-x/scale_factor)/2:
            x_pos = x
        else:
            x_pos = int(scale_factor*x_pos - (scale_factor*m_x-x)/2)
            
        if y_pos <= (m_y-y/scale_factor)/2:
            y_pos = r
        elif y_pos >= m_y-(m_y-y/scale_factor)/2:
            y_pos = y-r
        else:
            y_pos = int(scale_factor*y_pos-(scale_factor*m_y-y)/2)
            
        try:
            mask = np.zeros(img.shape[:2])
            mask = cv2.circle(mask,(x_pos,y_pos),r,color=(255,255,255),thickness=-1).astype(np.uint8)
            #image_patch = image[np.where(mask == 1)]
            image_patch = cv2.bitwise_and(img,img,mask=mask)
            d = image_patch.max(axis=(0,1))
            image = (255*(img*1.0 / d).clip(0, 1)).astype(np.uint8)
            return image
        except:
            print("Whitepatch Balancing error")
            return img
    
    def photo_image(self, image):
        height, width = image.shape[:2]
        if height > width:
            image = transform.rescale(image,(self.y_max-self.y_min)/max(image.shape),multichannel=True,anti_aliasing=True)
        else:
            image = transform.rescale(image,(self.x_max-self.x_min)/max(image.shape),multichannel=True,anti_aliasing=True)
        image = util.img_as_ubyte(image)
        image = PIL.Image.fromarray(image)
        return ImageTk.PhotoImage(image)
        
    def click(self, event):
        x = event.x
        y = event.y
        r = 7
        self.canvas1.coords('circle', x-r,y-r,x+r,y+r)
        self.update_right()
        
root = tk.Tk()
root.title('Cropping Utility')
root.resizable(False,False)
app = App(root)
root.mainloop()