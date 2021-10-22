from flask import Flask
from flask import request,jsonify,render_template
from flair.data import Sentence
from flair.models import SequenceTagger
import os
import requests
import shutil
from multiprocessing import Pool
import argparse
import imghdr
import base64
from pathlib import Path
import random
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from model import Model
from google_images_download import googleimagesdownload
import re
import cv2
import torch.backends.cudnn as cudnn
import time

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import gc

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])

app = Flask(__name__, static_folder='/')

texthistory = ""
nerhistory = ""
dlin = ""
dlresult = ""
slin = ""
slout = ""

def detect(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    
    count = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    if str(names[int(c)])=="person" and n == 1:
                        print(f'{s}Done. ({t2 - t1:.3f}s)')
                        return count

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
        count += 1

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    return 0

def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


@app.route('/')
def main():
    page=open('1.html',encoding='utf-8')
    res=page.read()
    return res

@app.route('/ner')
def ner():
    text = request.args.get("text")
    # make a sentence
    global texthistory
    global nerhistory
    if (text == texthistory):
        return nerhistory
    sentence = Sentence(text)
    texthistory = text

    # load the NER tagger
    tagger = SequenceTagger.load('ner')

    # run NER over sentence
    tagger.predict(sentence)
    a = dict()
    b = []
    # iterate over entities and print
    for entity in sentence.get_spans('ner'):
        s = ''
        if(entity.tag == 'PER'):
            for i in range(len(entity.tokens)):
                s += entity.tokens[i].text
                s += ' '
            s = s[:-1]
            if(s in a):
                a[s] += 1
            else:
                a[s] = 0
                if(len(entity.tokens)>1):
                    for i in range(len(entity.tokens)):
                        b.append(entity.tokens[i].text)
    c = []
    for i in a:
        for j in b:
            if(i == j):
                c.append(i)
    for i in c:
        a.pop(i)
    res = ''
    count = 0
    for i in a:
        count = count + 1
        if(count>5):
            break
        res = res + i + '@'
    nerhistory = res[:-1]
    return res[:-1]

@app.route('/download')
def dl():
    global dlin
    global dlresult
    keywords = request.args.get("keywords")
    title = request.args.get("title")
    number = int(request.args.get("number"))

    if (keywords == dlin):
        return dlresult
    keyword = keywords.split("@")
    dlin = keywords

    res = ""
    for i in range(len(keyword)):
        res = res + "<a href='http://202.120.38.146:9751/pic?keyword="+ keyword[i] + " in " + title + "&number="+str(number)+"'target='_blank'>" + keyword[i] +"</a>\n"

    response = googleimagesdownload()   #class instantiation

    for i in range(len(keyword)):
        key = keyword[i] + " in " + title
        Arg =  {"keywords":key,"limit":number,"color_type":"full-color"}
        path="./downloads/"+key
        if os.path.exists(path):
            continue
        paths = response.download(Arg)   #passing the arguments to the function
        name="google_000"
        fileType=".jpg"
        count=0
        filelist=os.listdir(path)
        for files in filelist:
            Olddir=os.path.join(path,files)
            if os.path.isdir(Olddir):
                continue
            Newdir=os.path.join(path,name+str(count)+fileType)
            os.rename(Olddir,Newdir)
            count+=1
    
        width=300
        depth=300
        path=path+"/"
        l=os.listdir(path)
        for pic in l:
            im=Image.open(path+pic)
            w,h=im.size
            if im.mode == "P" or im.mode == "RGBA":
                im = im.convert('RGB')

            if w>=h:
                h_new=int(width*h/w)
                w_new=width
                out = im.resize((w_new,h_new),Image.ANTIALIAS)
                out.save(path+pic)
            else :
                w_new=int(depth*w/h)
                h_new=depth
                out = im.resize((w_new,h_new),Image.ANTIALIAS)
                out.save(path+pic)

            size = 1024 * 10
            im = Image.open(path+pic)
            size_tmp = os.path.getsize(path+pic)
            q = 100
            while size_tmp > size and q > 0:
                out = im.resize(im.size, Image.ANTIALIAS)
                out.save(path+pic, quality=q)
                size_tmp = os.path.getsize(path+pic)
                q -= 5
                
    # for i in range(len(keyword)):
    #     key = keyword[i]
    #     Arg =  {"keywords":key,"limit":number}
    #     path="./downloads/"+key
    #     if os.path.exists(path):
    #         continue
    #     paths = response.download(Arg)   #passing the arguments to the function
    #     name="google_000"
    #     fileType=".jpg"
    #     count=0
    #     filelist=os.listdir(path)
    #     for files in filelist:
    #         Olddir=os.path.join(path,files)
    #         if os.path.isdir(Olddir):
    #             continue
    #         Newdir=os.path.join(path,name+str(count)+fileType)
    #         os.rename(Olddir,Newdir)
    #         count+=1
    
    #     width=300
    #     depth=300
    #     path=path+"/"
    #     l=os.listdir(path)
    #     for pic in l:
    #         im=Image.open(path+pic)
    #         w,h=im.size
    #         if im.mode == "P" or im.mode == "RGBA":
    #             im = im.convert('RGB')

    #         if w>=h:
    #             h_new=int(width*h/w)
    #             w_new=width
    #             out = im.resize((w_new,h_new),Image.ANTIALIAS)
    #             out.save(path+pic)
    #         else :
    #             w_new=int(depth*w/h)
    #             h_new=depth
    #             out = im.resize((w_new,h_new),Image.ANTIALIAS)
    #             out.save(path+pic)

    #         size = 1024 * 10
    #         im = Image.open(path+pic)
    #         size_tmp = os.path.getsize(path+pic)
    #         q = 100
    #         while size_tmp > size and q > 0:
    #             out = im.resize(im.size, Image.ANTIALIAS)
    #             out.save(path+pic, quality=q)
    #             size_tmp = os.path.getsize(path+pic)
    #             q -= 5
    dlresult = res
    return res

@app.route('/pic')
def pic():
    keywords = request.args.get("keyword")
    number = int(request.args.get("number"))
    # keyword = keywords.split()
    # page=open('pic.html?keyword='+keyword[0]+'%20'+keyword[1]+'%20'+keyword[2],encoding='utf-8')
    # res=page.read()
    # return keyword
    return render_template('./pic.html', keyword = keywords,number = number)

@app.route('/select')
def select():
    global slin
    global slout
    keywords = request.args.get("keywords")
    title = request.args.get("title")
    keyword = keywords.split("@")
    index = ''
    if (keywords == slin):
        return slout

    slin = keywords
    for i in range(len(keyword)):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov5x.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='downloads/'+keyword[i] + " in " + title, help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        opt = parser.parse_args()
        check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

        res = detect(opt=opt)
        index += "./downloads/"+ keyword[i] + " in " + title +"/google_000"+str(res)+".jpg@"
    slout = index[:-1]
    return index[:-1]

@app.route('/style')
def style():
    ix = request.args.get("index")
    stindex = int(request.args.get("style"))
    index = ix.split("@")
    st = ""
    if stindex == 1:
        st = "./style/sketch.png"
    if stindex == 2:
        st = "./style/trial.jpg"
    if stindex == 3:
        st = "./style/picasso_self_portrait.jpg"

    for i in range(len(index)):
        parser = argparse.ArgumentParser(description='Style Swap by Pytorch')
        parser.add_argument('--content', '-c', type=str, default=index[i],
                            help='Content image path e.g. content.jpg')
        parser.add_argument('--style', '-s', type=str, default=st,
                            help='Style image path e.g. image.jpg')
        parser.add_argument('--output_name', '-o', type=str, default=index[i]+str(stindex),
                            help='Output path for generated image, no need to add ext, e.g. out')
        parser.add_argument('--alpha', '-a', type=float, default=1,
                        help='alpha control the fusion degree in Adain')
        parser.add_argument('--gpu', '-g', type=int, default=0,
                            help='GPU ID(nagative value indicate CPU)')
        parser.add_argument('--model_state_path', type=str, default='model_state.pth',
                            help='save directory for result and loss')

        args = parser.parse_args()

        # set device on GPU if available, else CPU
        if torch.cuda.is_available() and args.gpu >= 0:
            device = torch.device(f'cuda:{args.gpu}')
            print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
        else:
            device = 'cpu'

        # set model
        model = Model()
        if args.model_state_path is not None:
            model.load_state_dict(torch.load(args.model_state_path, map_location=lambda storage, loc: storage))
        model = model.to(device)

        c = Image.open(args.content)
        if c.mode!='RGB':
            c = c.convert("RGB")
        s = Image.open(args.style)
        c_tensor = trans(c).unsqueeze(0).to(device)
        s_tensor = trans(s).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.generate(c_tensor, s_tensor, args.alpha)
        
        out = denorm(out, device)

        if args.output_name is None:
            c_name = os.path.splitext(os.path.basename(args.content))[0]
            s_name = os.path.splitext(os.path.basename(args.style))[0]
            args.output_name = f'{c_name}_{s_name}'

        save_image(out, f'{args.output_name}.jpg', nrow=1)
        # o = Image.open(f'{args.output_name}.jpg')

        # demo = Image.new('RGB', (c.width * 2, c.height))
        # o = o.resize(c.size)
        # s = s.resize((i // 4 for i in c.size))

        # demo.paste(c, (0, 0))
        # demo.paste(o, (c.width, 0))
        # demo.paste(s, (c.width, c.height - s.height))
        # demo.save(f'{args.output_name}_style_transfer_demo.jpg', quality=95)

        # o.paste(s,  (0, o.height - s.height))
        # o.save(f'{args.output_name}_with_style_image.jpg', quality=95)

        print(f'result saved into files starting with {args.output_name}')
        gc.collect()
        torch.cuda.empty_cache()
    return ix

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=9751)