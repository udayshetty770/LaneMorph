import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import time

@torch.no_grad()
def run(weights=ROOT / 'toys.pt',  # model.pt path(s)
        source=ROOT / '1',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.30,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    print(names)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    global m
    m = 1
    global np
    np = 1
    count = 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        road1=0
        road2=0
        road3=0
        road4=0
        counting=0
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            im0 = cv2.resize(im0, (640, 640))
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            height, width, _ = im0.shape

            gridx1 = int(width/3)
            gridx2 = gridx1 + gridx1

            gridy1 = int(height/3)
            gridy2 = gridy1 + gridy1

            cv2.line(im0, (gridx1, 0), (gridx1, height), (0, 255, 0), 1)
            cv2.line(im0, (gridx2, 0), (gridx2, height), (0, 255, 0), 1)
            cv2.line(im0, (0, gridy1), (width, gridy1), (0, 255, 0), 1)
            cv2.line(im0, (0, gridy2), (width, gridy2), (0, 255, 0), 1)

            Ambulance_detection = 0
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        counting += 1
                        annotator.box_label(xyxy, names[c], color=colors(c, True))

                        if names[c] == 'Ambulance':
                            Ambulance_detection = 1

                        x, y, w, h = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                        if x > gridx1 and x < gridx2 and y < gridy1:
                            road1 +=1
                            
                        if x < gridx2 and y > gridy1 and y < gridy2:
                            
                            road2 +=1
                            
                        if x > gridx1 and x > gridx2 and y < gridy2:
                            
                            road3 +=1
                            
                        if x < gridx1 and y > gridy1 and y < gridy2:
                            
                            road4 +=1
                                
                print('Total Vehicles in road1 {}'.format(road1))
                print('Total Vehicles in road2 {}'.format(road2))
                print('Total Vehicles in road3 {}'.format(road3))
                print('Total Vehicles in road4 {}'.format(road4))
                count1='{}{}{}{}'.format(road1, road2, road3, road4)
                print(count1)
                print('============================================')

                if Ambulance_detection == 1:
                    cv2.circle(im0, (gridx1+25, 50), 15, (0, 255, 0), -1)
                    cv2.putText(im0, str(road1 * 10), (gridx1, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) , 2)
                    cv2.circle(im0, (gridx2, gridy1), 15, (0, 0, 255), -1)
                    cv2.putText(im0, str(road2 * 10), (gridx2, gridy1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) , 2)
                    cv2.circle(im0, (gridx1, gridy2), 15, (0, 0, 255), -1)
                    cv2.putText(im0, str(road3 * 10), (gridx1, gridy2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) , 2)
                    cv2.circle(im0, (50, gridy1), 15, (0, 0, 255), -1)
                    cv2.putText(im0, str(road4 * 10), (50, gridy1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) , 2)
                else:

                    counts = [road1, road2, road3, road4]
                    labels = ['a', 'b', 'c', 'd']
                    sorted_counts, sorted_labels = zip(*sorted(zip(counts, labels), reverse=True))
                    sorted_counts = list(sorted_counts)
                    sorted_labels = list(sorted_labels)

                    print(sorted_counts)  # Output: [4, 2, 1, 0]
                    print(sorted_labels)  # Output: ['c', 'a', 'b', 'd']
                    rgb_colors = [
                        (0, 255, 0),   # Green
                        (0, 255, 255), # Yellow
                        (0,165, 255), # Orange
                        (0, 0, 255)    # Red
                    ]

                    for i in range(4):
                        if sorted_labels[i] == 'a':
                            cv2.circle(im0, (gridx1+25, 50), 15, rgb_colors[i], -1)
                            cv2.putText(im0, str(road1 * 10), (gridx1, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_colors[i] , 2)
                        if sorted_labels[i] == 'b':
                            cv2.circle(im0, (gridx2, gridy1), 15, rgb_colors[i], -1)
                            cv2.putText(im0, str(road2 * 10), (gridx2, gridy1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_colors[i] , 2)
                        if sorted_labels[i] == 'c':
                            cv2.circle(im0, (gridx1, gridy2), 15, rgb_colors[i], -1)
                            cv2.putText(im0, str(road3 * 10), (gridx1, gridy2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_colors[i] , 2)
                        if sorted_labels[i] == 'd':
                            cv2.circle(im0, (50, gridy1), 15, rgb_colors[i], -1)
                            cv2.putText(im0, str(road4 * 10), (50, gridy1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_colors[i] , 2)
                
            # Stream results
            im0 = annotator.result()
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'toys.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.80, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":  
    opt = parse_opt()
    main(opt)
