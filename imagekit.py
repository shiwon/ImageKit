import PySimpleGUI as sg
from PIL import Image, ImageTk
import io
import os
import glob
import math

import numpy as np
from torchvision.transforms.functional import to_tensor
from pytorch_msssim import ssim, ms_ssim
import lpips
lpips_loss = lpips.LPIPS(net="vgg")

def get_lpips(path1,path2):
    org=to_tensor(Image.open(path1).convert('L')).unsqueeze(0)
    dist=to_tensor(Image.open(path2).convert('L')).unsqueeze(0)
    # Higher means further/more different. Lower means more similar.
    return lpips_loss(dist, org).item()

def get_ms_ssim(path1,path2):
    org=to_tensor(Image.open(path1).convert('L')).unsqueeze(0)
    dist=to_tensor(Image.open(path2).convert('L')).unsqueeze(0)
    return ms_ssim(org,dist,data_range=1).item()

def get_psnr_Y(path1,path2):
    print(path1,path2)
    org=Image.open(path1).convert('L')
    dist=Image.open(path2).convert('L')
    w,h=org.size
    
    pixel_value_Ori=np.array(org).flatten().astype(float)
    pixel_value_Dis=np.array(dist).flatten().astype(float)

    addr = h*w
    sum=0
    for i in range(addr):
        sum += (pixel_value_Ori[i]-pixel_value_Dis[i])*(pixel_value_Ori[i]-pixel_value_Dis[i])

    MSE = sum / addr
    PSNR = 10 * math.log(255*255/MSE,10)
    return PSNR

def calculate_metrics_from_directory(ref,path):
    refs =glob.glob(ref+'/*')
    paths=glob.glob(path+'/*')
    refs.sort()
    paths.sort()
    length=len(refs)
    sum_psnr=0
    sum_ms_ssim=0
    sum_lpips=0
    if length==len(paths):
        for i in range(length):
            sum_psnr+=get_psnr_Y(refs[i],paths[i])
            sum_ms_ssim+=get_ms_ssim(refs[i],paths[i])
            sum_lpips+=get_lpips(refs[i],paths[i])
            #print('result: '+str(psnr)+','+str(ssim),','+str(lpips_val))
    psnr_ave=sum_psnr/length
    ms_ssim_ave=sum_ms_ssim/length
    lpips_ave=sum_lpips/length
    print(f'psnr:{psnr_ave},ms-ssim:{ms_ssim_ave},lpips:{lpips_ave}')
    return psnr_ave,ms_ssim_ave,lpips_ave

def get_img_data(f, maxsize=(600, 450),first=False,start_point=None,size=None):
    """Generate image data using PIL"""
    if start_point and size:
        img=get_crop_image(f,start_point,size)
        w,h=maxsize
        w_ratio=w/img.width
        h_ratio=h/img.height
        dstsize=(w,round(img.height*w_ratio)) if w_ratio<h_ratio else (round(img.width * h_ratio), h)
        img=img.resize(dstsize)
    else:
        img = Image.open(f)
        img.thumbnail(maxsize)

    if first:  # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)

def get_scale_factor(org_size,max_size):
    org_w,org_h=org_size
    dst_w,dst_h=max_size
    f_w=org_w/dst_w
    f_h=org_h/dst_h
    f=max(f_w,f_h)
    return f if f>1 else 1

def get_crop_image(src_path,start_point=(0,0),size=(100,100)):
    src=Image.open(src_path)
    dst=src.crop((start_point[0],start_point[1],start_point[0]+size[0],start_point[1]+size[1]))
    return dst

import re
class ResizeWindow:
    def __init__(self,src_path):
        self.src_path=src_path
        self.src_image=Image.open(src_path)
        self.w,self.h=self.src_image.size
        self.view_size=(400,400)
        self.layout=self.set_layout()
        self.window=sg.Window('resize window', self.layout, return_keyboard_events=True)
    def set_layout(self):
        col_input_size=[sg.Text(f'input image     width:{self.w} height:{self.h}')]
        col_output_size=[sg.Text('output image     width:'),sg.InputText(size=(5,1),enable_events=True ,key='-WIDTH-'),sg.Text('height:'),sg.InputText(size=(5,1),enable_events=True ,key='-HEIGHT-')]
        col_buttons=[sg.Button('save',key='-SAVE-'),sg.Button('cancel',key='-CANCEL-')]
        
        self.view_elem=sg.Image(data=get_img_data(self.src_path,self.view_size,True),key='-IMAGE-')
        col_view=[self.view_elem]

        left_cols=sg.Column([col_input_size,col_output_size,col_buttons])
        right_cols=sg.Column([col_view])
        
        return [[left_cols,right_cols]]
    def update_view(self):
        view_image = self.src_image.resize((self.w,self.h))
        view_image.thumbnail(self.view_size)
        bio = io.BytesIO()
        view_image.save(bio, format="PNG")
        del view_image
        self.view_elem.update(data=bio.getvalue())
    
    def popup(self):
        while True:
            event,values=self.window.read()

            if event in ('-CANCEL-',None):
                break
            elif event=='-WIDTH-':
                if not (values['-WIDTH-'].isdecimal() and values['-WIDTH-'].isascii()):
                    self.window['-WIDTH-'].update(re.sub(r'\D','',values['-WIDTH-']))
                else:
                    self.w=int(values['-WIDTH-'])
                    self.update_view()
            elif event=='-HEIGHT-':
                if not (values['-HEIGHT-'].isdecimal() and values['-HEIGHT-'].isascii()):
                    self.window['-HEIGHT-'].update(re.sub(r'\D','',values['-HEIGHT-']))
                else:
                    self.h=int(values['-HEIGHT-'])
                    self.update_view()
            elif event=='-SAVE-':
                if self.w and self.h:
                    default_path=f'{os.path.dirname(self.src_path)}/resized_{os.path.basename(self.src_path)}'
                    dstpath=sg.popup_get_file('save',save_as=True,default_path=default_path)
                    self.src_image.resize((self.w,self.h)).save(dstpath)
                    break
        self.window.close()




if __name__=="__main__":
    metric_result_path='metrics.csv'
    gt_graph_size=(576,324)
    lr_graph_size=(280,280)
    scale_factor=1
    gt_filename=lr_filename=None
    all_filenames=["",""]
    opt_filenames=[]
    gt_graph_elem=sg.Graph(
        canvas_size=gt_graph_size,
        graph_bottom_left=(0, 0),
        graph_top_right=gt_graph_size,
        key="-GT-GRAPH-",
        change_submits=True,  # mouse click events
        background_color='lightblue',
        drag_submits=True)
    croped_gt_graph_elem=sg.Graph(
        canvas_size=lr_graph_size,
        graph_bottom_left=(0, 0),
        graph_top_right=lr_graph_size,
        key="-CROP-GT-GRAPH-",
        change_submits=True,  # mouse click events
        #background_color='gray',
        drag_submits=True)
    croped_lr_graph_elem=sg.Graph(
        canvas_size=lr_graph_size,
        graph_bottom_left=(0, 0),
        graph_top_right=lr_graph_size,
        key="-CROP-LR-GRAPH-",
        change_submits=True,  # mouse click events
        #background_color='gray',
        drag_submits=True)
    col_image=[gt_graph_elem]
    col_image_window=[croped_gt_graph_elem,croped_lr_graph_elem]
    col_read_file=[sg.InputText('ファイルを選択', key='-INPUT-TEXT-', enable_events=True, ),
                 sg.FileBrowse('Ground Truth 画像を読み込む', key='-GT-FILE-',target='-INPUT-TEXT-',file_types=(('png', '*.png'),)),
                 sg.Button('選択した範囲にクロッピング',key='-CROP-'),
                 sg.Button('リサイズして保存',key='-RESIZE-'),
                 sg.Button('メトリックの計算',key='-CALC-')]
    col_read_file_lr=[sg.InputText('ファイルを選択', key='-LR-INPUT-TEXT-', enable_events=True, ),
                 sg.FileBrowse('distorted 画像を読み込む', key='-LR-FILE-',target='-LR-INPUT-TEXT-',file_types=(('png', '*.png'),))]
    col_read_file_lr2=[sg.InputText('ファイルを選択', key='-LR-INPUT-TEXT2-', enable_events=True, ),
                 sg.FileBrowse('追加する', key='-ADD-LR-FILE-',target='-LR-INPUT-TEXT2-',file_types=(('png', '*.png'),))]
    col_crop=[sg.Text('rect size:'),sg.Text(0,key='-CROP-WIDTH-'),sg.Text('x'),sg.Text(0,key='-CROP-HEIGHT-')]
    
    layout=[col_read_file,
    col_read_file_lr,
    col_read_file_lr2,
    col_image,
    col_crop,
    col_image_window,
    [sg.Text(key='-ALL-FILENAME-')]]

    window = sg.Window('image kit', layout, return_keyboard_events=True, location=(0, 0), use_default_focus=False)
    gt_graph = window["-GT-GRAPH-"]
    cr_gt_graph = window["-CROP-GT-GRAPH-"]
    cr_lr_graph = window["-CROP-LR-GRAPH-"]
    dragging = False
    start_point = end_point = prior_rect = None
    pil_start_point=None
    rect_size=(0,0)
    while True:
        # read the form
        event, values = window.read()
        # perform button and keyboard operations
        if event is None:
            break
        elif event == "-GT-GRAPH-":  # if there's a "Graph" event, then it's a mouse
            x, y = values["-GT-GRAPH-"]
            if not dragging:
                start_point = (x, y)
                dragging = True
            else:
                c_w=window['-CROP-WIDTH-']
                c_h=window['-CROP-HEIGHT-']
                rect_size=(scale_factor*abs(x-start_point[0]),scale_factor*abs(y-start_point[1]))
                c_w.update(int(rect_size[0]))
                c_h.update(int(rect_size[1]))
                end_point = (x, y)
            if prior_rect:
                gt_graph.delete_figure(prior_rect)
            if None not in (start_point, end_point):
                prior_rect = gt_graph.draw_rectangle(start_point, end_point, line_color='red')
        elif event.endswith('+UP') and start_point and end_point and rect_size[0]>1 and rect_size[1]>1:  # The drawing has ended because mouse up
            sx = min(start_point[0],end_point[0])
            sy = gt_graph_size[1]-max(start_point[1],end_point[1])
            pil_start_point=(scale_factor*sx,scale_factor*sy)
            if lr_filename:
                cr_lr_graph.erase()
                cr_gt_graph.erase()
                cr_lr_graph.draw_image(data=get_img_data(lr_filename,lr_graph_size,True,pil_start_point,rect_size), location=(0,lr_graph_size[1]))
                cr_gt_graph.draw_image(data=get_img_data(gt_filename,lr_graph_size,True,pil_start_point,rect_size), location=(0,lr_graph_size[1]))
            start_point, end_point = None, None  # enable grabbing a new rect
            dragging = False
        elif event == '-CROP-':
            if gt_filename and os.path.isfile(gt_filename):
                dirname=os.path.basename(os.path.dirname(gt_filename))
                dstpath=f'crops/{dirname}.png'
                get_crop_image(gt_filename,pil_start_point,rect_size).save(dstpath)
            if lr_filename and os.path.isfile(lr_filename):
                dirname=os.path.basename(os.path.dirname(lr_filename))
                dstpath=f'crops/{dirname}.png'
                get_crop_image(lr_filename,pil_start_point,rect_size).save(dstpath)
            if opt_filenames:
                for fname in opt_filenames:
                    dirname=os.path.basename(os.path.dirname(fname))
                    dstpath=f'crops/{dirname}.png'
                    get_crop_image(fname,pil_start_point,rect_size).save(dstpath)
            print(f'start:{pil_start_point},width:{rect_size[0]},height:{rect_size[1]}')
        elif event == '-RESIZE-':
            if gt_filename and os.path.isfile(gt_filename):
                r_window=ResizeWindow(gt_filename)
                r_window.popup()
                del r_window
                print('save gt file')
        elif event == '-CALC-':
            with open(metric_result_path,"w") as f:
                f.write(",PSNR,MS-SSIM,LPIPS\n")
                if gt_filename and os.path.isfile(gt_filename):
                    gt_dir=os.path.dirname(gt_filename)
                    gt_basedir=os.path.basename(gt_dir)
                    if lr_filename and os.path.isfile(lr_filename):
                        lr_dir=os.path.dirname(lr_filename)
                        lr_basedir=os.path.basename(lr_dir)
                        psnr_ave,ms_ssim_ave,lpips_ave=calculate_metrics_from_directory(gt_dir,lr_dir)
                        f.write(f'{lr_basedir},{psnr_ave},{ms_ssim_ave},{lpips_ave}\n')
                    if opt_filenames:
                        for fname in opt_filenames:
                            lr_dir=os.path.dirname(fname)
                            lr_basedir=os.path.basename(lr_dir)
                            psnr_ave,ms_ssim_ave,lpips_ave=calculate_metrics_from_directory(gt_dir,lr_dir)
                            f.write(f'{lr_basedir},{psnr_ave},{ms_ssim_ave},{lpips_ave}\n')

        elif values['-GT-FILE-'] != '':
            gt_filename=values['-INPUT-TEXT-']
            if os.path.isfile(gt_filename):
                gt_graph.erase()
                gt_graph.draw_image(data=get_img_data(gt_filename,gt_graph_size,first=True), location=(0,gt_graph_size[1]))
                scale_factor=get_scale_factor(Image.open(gt_filename).size,gt_graph_size)
                dragging = False
                start_point = end_point = prior_rect = None
                pil_start_point=None
                rect_size=(0,0)
                window['-CROP-WIDTH-'].update(0)
                window['-CROP-HEIGHT-'].update(0)
                all_filenames[0]=gt_filename
        if values['-LR-FILE-'] != '':
            if not gt_filename:
                sg.popup_error('先にgt画像を設定してください')
            else:
                lr_filename=values['-LR-INPUT-TEXT-']
                all_filenames[1]=lr_filename
        if values['-ADD-LR-FILE-'] != '':
            if not gt_filename:
                sg.popup_error('先にgt画像を設定してください')
            if not lr_filename:
                sg.popup_error('1枚目のlr画像を設定してください')
            else:
                if not values['-LR-INPUT-TEXT2-'] in opt_filenames:
                    opt_filenames.append(values['-LR-INPUT-TEXT2-'])
        
        display_filename=all_filenames+opt_filenames
        window['-ALL-FILENAME-'].update(display_filename)