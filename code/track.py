import argparse
import cv2
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

# # 变量初始化
# ——@@@@@ 新加的代码 @@@@@——
# 变量初始化，车数和每个车的id组成的列表
count = 0
obj_id_list = []
# 开关决定是否采用 过线计数的方法
is_count_by_line = True
# ——@@@@@ 新加的代码（1） @@@@@——

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov8n.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu

        # 原来的
        # show_vid=False,  # show results
        # 修改以后的
        show_vid=True,  # show results

        # 原来的
        # save_txt=False,  # save results to *.txt
        # 修改后
        save_txt=True, # save results to *.txt

        save_conf=False,  # save confidences in --save-txt labels

        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track


        save_vid=False,  # save confidences in --save-txt labels


        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    # Dataloader
    bs = 1
    if webcam:
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams( # 加载视频
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        bs = len(dataset)
    else:
        dataset = LoadImages( # 加载图片
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs

    # Run tracking
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            preds = model(im, augment=augment, visualize=visualize)

        # Apply NMS
        with dt[2]:
            if is_seg:
                masks = []

                # 原来的
                # p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                # 新改的（因为多了个nm这里报错了,所以给删了，而且non_max_suppression的官方没有提供nm这个参数）
                p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


                proto = preds[1][-1]
            else:
                p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
        # Process detections
        for i, det in enumerate(p):  # detections per image
            seen += 1
            if webcam:  # bs >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                if is_seg:
                    shape = im0.shape
                    # scale bbox first the crop masks
                    if retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                        masks.append(process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2]))  # HWC
                    else:
                        masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                else:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                
                # draw boxes for visualization;画出bouding box
                if len(outputs[i]) > 0:
                    
                    if is_seg:
                        # Mask plotting
                        annotator.masks(
                            masks[i],
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                            255 if retina_masks else im[i]
                        )
                    
                    for j, (output) in enumerate(outputs[i]):
                        
                        bbox = output[0:4]
                        id = output[4] # 识别出的object的id
                        cls = output[5]  # 识别出的object的类别
                        conf = output[6]

                        # 完成计数
                        # ——@@@@@ 新添加的代码（2） @@@@@——
                        w, h = im0.shape[1], im0.shape[0] # 获得汽车目标这个小图的宽和高
                        # 根据你规定的函数进行计数
                        # ——bbox是识别出来的小车的方框的 四个数字表示坐标
                        # ——w 和 h是视频每一帧图片的长和宽

                        # 计数的时候启用哪种，可以做个判断。判断使用哪个函数进行计数
                        if is_count_by_line:
                            # print("id is ",id)
                            # print("Video window size is %s pixels width and %s pixels height"%(w,h))
                                # object中心点过了低端30%才算
                            # count_obj_cross_line_center(bbox, w, h, id)  # 中心过线才计入数字
                            count_obj_cross_line_any(bbox, w, h, id) # 纵向有一点点过线就计入数字
                            # count_obj_all(id)  # 只要出现，不重复就计入数字

                            # ——@@@@@ 新添加的代码(2) @@@@@——
                        else:
                            # ——@@@@@ 新加的代码（9） @@@@@——
                            count_obj_all(id)
                            # ——@@@@@ 新加的代码（9） @@@@@——

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            x1 = (output[0] + output[2]) / 2
                            print(x1)
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            color = colors(c, True)
                            annotator.box_label(bbox, label, color=color)
                            
                            if save_trajectories and tracking_method == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(np.array(bbox, dtype=np.int16), imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                            
            else:
                # pass

                #tracker_list[i].tracker.pred_n_update_all_tracks() # 这是作者注释掉的，不是我注释掉的

                # 防止后面因为参数未定义引发的报错
                # ——@@@@@ 新添加的代码（7） @@@@@——
                # 当视频刚刚开始识别的时候，画面中没有一辆车，所以line252这个判断无法通过“ if len(outputs[i]) > 0:”
                # 所判断为True里面，line272的 “w, h = im0.shape[1], im0.shape[0]”，自然就没有执行，所以w 和 h就没有定义，会报错
                # 为了不报这个错，这里定义一次。。后面画面中有车了以后走 true那个部分也就不会重复定义了
                w, h = im0.shape[1], im0.shape[0]
                # ——@@@@@ 新添加的代码（7） @@@@@——

            #

            w, h = im0.shape[1], im0.shape[0]
            # Stream results
            im0 = annotator.result()
            if show_vid:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])


                # 每帧图片累计汽车的数字显示在左上角
                # ——@@@@@ 新添加的代码（3） @@@@@——
                # ——用于在左上角显示每一帧累计通过的汽车输出量
                global count # 把全局变量中那个count的数据拿过来这里要显示在窗口里
                org = (150, 150) # 数字的大小吧，我猜
                font = cv2.FONT_HERSHEY_SIMPLEX # 字体
                fontScale = 3 # 字号
                color = (0, 255, 0)
                thickness = 3 # 厚度？
                cv2.putText(im0, str(count), org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
                # ——@@@@@ 新添加的代码 @@@@@——


                # 给在视频每一帧的图片的最先有这样一条绿色的横线
                # ——@@@@@ 新添加的代码（6） @@@@@——

                # 让横线距离最底端留个350像素，是仅仅适用于Traffic.mp4的那个画面的尺寸的。但是有些视频的尺寸很小，你留个350像素很多横线就到画面上面的外面了
                # 之前的
                # start_point = (0, h-350) # 横坐标顶着照片的最左边，纵坐标是照片的高度 减去  底下空出来的350个像素
                # 之后修改的
                # ——之所以这里加了一个int,是因为后面输入的这个cv2.Line()函数的输入值必须是整数，但是你乘以一个0.7就转成float了,所以要转回int
                start_point = (0, int(h-0.3*h)) # 距离照片最底端留下30%

                # 之前的
                # end_point = (w, h-350) # 横坐标顶着照片的最右边，纵坐标，同上
                # 之后修改的
                end_point = (w, int(h-0.3*h)) # 横坐标顶着照片的最右边，纵坐标，同上


                # 是否加横着的绿色的线，做个判断
                # ——是过线才计数，你才有必要划线啊！
                if is_count_by_line:
                    cv2.line(im0, start_point, end_point, color, thickness=2) # 非过线的方法，就注释掉这句
                    # ——@@@@@ 新添加的代码（6） @@@@@——


                cv2.imshow(str(p), im0) # 将图片上加的这些东西和每一帧图片本身都显示出来
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
            
        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)

    # 将人数和id列表打印出来
    # ——@@@@@ 新加的代码（5） @@@@@——
    print("*"*50)
    print("There are %s objects in total"%(count))
    print("The id list : %s"%(obj_id_list))
    print("*" * 50)
    # ——@@@@@ 新加的代码（5） @@@@@——

# 定义实现计数的函数
# ——@@@@@ 新加的代码（4） @@@@@——
def count_obj_cross_line_center(box,w,h,id): # ——@@@@@ 修该过（7） @@@@@——
                                      # 特别强调这是跨越线的
      # ——bbox是识别出来的小车的方框的 四个数字表示坐标
      # ——w 和 h是视频每一帧图片的长和宽
    global count,obj_id_list
    center_coordinates = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))
        # 分别求出识别方框的横纵坐标的中心点，从拿到目标的中心点

        # 如过目标的纵坐标大于图片的高减去350个像素（也就是绿色的线所在的位置）（图片的坐标是从左上角为中心，向下向右为正半轴）
        # ——整个图片的高度为h,350是绿色线距离最底端的距离
        # ——这个车的左上角的纵坐标是box[3],整个车的纵坐标的中心点是 box[3]+车高的一半，也就是下面这个柿子
        # 也就是目标的“纵坐标中心点”低于这条横着的绿色线，并且这个目标之前没有出现，就将id记录进data,总车数加1
    # 原来的
    # if int(box[1]+(box[3]-box[1])/2) > (h -350):
    # 修改后的
    if int(box[1]+(box[3]-box[1])/2) > (int(h-0.3*h)): # 低于70%就计数
        if  id not in obj_id_list:
            count += 1
            obj_id_list.append(id)

    # print('我猜是横坐标',box[0],'\t',box[2]) # 横坐标
    # print('我猜是纵坐标',box[1],'\t',box[3]) # 纵坐标
    # print("#"*20)
# ——@@@@@ 新加的代码（4） @@@@@——


# ——@@@@@ 新加的代码（4） @@@@@——
    # 只要object纵向上有一点碰到了低端的30%,就count进去
def count_obj_cross_line_any(box,w,h,id):
    global count,obj_id_list
    y_upper = box[1] # object的上端纵坐标
    y_lower = box[3] # object的下端纵坐标

    # 这个行人可以从下端进来，y_upper先大于 低端30%的纵坐标；行人也有可能从上端进来，即y_lower大于低端30%的坐标
    # 两个条件满足其中之一即可，所以用or
    # 行人从远处走近的可能性比从镜头后面走入然后进入画面的可能性高,所以放在or前面
    if (y_lower > int(h-0.3*h)) or (y_upper > int(h-0.3*h)): # 低于70%就计数
        if  id not in obj_id_list:
            count += 1
            obj_id_list.append(id)




# 新增一个计数的函数，只要id不重复就计入数字
def count_obj_all(id):
    global count, obj_id_list  # 不global的话，就无法修改全局变量
    if id not in obj_id_list:
        count += 1
        obj_id_list.append(id)

def parse_opt():
    parser = argparse.ArgumentParser()

    # 改之前，识别和分割一起做，model size=23.9MB
    # parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov8s-seg.pt', help='model.pt path(s)')
    # 改成只做识别的模型，model size=6.5MB
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'best.pt', help='model.pt path(s)')

    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='C:\\Users\\yangyx\\Desktop\\1.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # 原来的
    # parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    # 更改以后
    parser.add_argument('--show-vid', default=True,action='store_true', help='display tracking video results') # 在运行程序的过程中，把视频中识别出来的方框打上，方框跟着车跑

    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt') # 是否把每一帧图像的识别结果都保存在txt文件
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') # 是否在txt文件保存置信度这个数据
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes') # 我猜应该是是否保存每个识别的方框的截图到本地
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track') # 是否记录每次追踪的轨迹线
    # 修改前
    # parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    # 修改后
    parser.add_argument('--save-vid', default=True,action='store_true', help='save video tracking results') # 把追踪视频的录像保存成文件
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3') # 只识别特定的类别，减少分类的任务负担
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name') # 识别结果文件保存的位置./runs/track
    parser.add_argument('--name', default='exp', help='save results to project/name') # 保存文件新建文件夹的名字是exp
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)') # 方框 bounding box的厚度
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt
def main(opt):
    #check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
