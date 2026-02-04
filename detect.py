import argparse
import os
import platform
import sys
from pathlib import Path

import torch

# Import necessary YOLO components
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadScreenshots
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# Dictionary mapping class names to descriptions and uses
class_descriptions = {

    "person": "A human being. Uses: security, crowd monitoring, social interaction.",
    "bicycle": "A human-powered vehicle with two wheels. Uses: transportation, exercise.",
    "car": "A four-wheeled motor vehicle. Uses: personal and commercial transportation.",
    "motorcycle": "A two-wheeled motor vehicle. Uses: transportation, recreation.",
    "airplane": "A powered flying vehicle with wings. Uses: air travel, cargo transport.",
    "bus": "A large motor vehicle for transporting passengers. Uses: public transport, school transport.",
    "train": "A series of connected vehicles that run on a track. Uses: long-distance travel, cargo transport.",
    "truck": "A motor vehicle designed primarily for carrying cargo. Uses: freight transport, construction.",
    "traffic light": "A signaling device for road traffic. Uses: traffic control, safety.",
    "fire hydrant": "A connection point for firefighters. Uses: fire suppression.",
    "stop sign": "A regulatory sign to stop vehicles. Uses: traffic control.",
    "parking meter": "A device that collects fees for parking. Uses: parking management.",
    "bench": "A long seat for multiple people. Uses: public seating, parks.",
    "bird": "A flying animal. Uses: ecological monitoring, avian studies.",
    "cat": "A small domesticated carnivorous mammal. Uses: companionship, pest control.",
    "dog": "A domesticated carnivorous mammal. Uses: companionship, service animals.",
    "horse": "A large domesticated mammal. Uses: transportation, agriculture.",
    "sheep": "A domesticated ruminant animal. Uses: wool, meat, dairy.",
    "cow": "A large domesticated ungulate. Uses: dairy products, meat.",
    "elephant": "A large land mammal. Uses: conservation, tourism.",
    "giraffe": "A tall African mammal with a long neck. Uses: wildlife tourism, education.",
    "zebra": "A wild equine with black and white stripes. Uses: wildlife conservation, tourism.",
    "carrot": "An orange root vegetable. Uses: food, agriculture.",
    "broccoli": "A green vegetable. Uses: food, health.",
    "apple": "A round fruit. Uses: food, agriculture.",
    "banana": "A long, curved fruit. Uses: food, nutrition.",
    "orange": "A round citrus fruit. Uses: food, vitamin C source.",
    "tree": "A perennial plant with an elongated stem. Uses: oxygen production, shade, lumber.",
    "bush": "A shrub. Uses: landscaping, wildlife habitat.",
    "flower": "A reproductive structure of flowering plants. Uses: decoration, gardening.",
    "knife": "A cutting tool. Uses: food preparation, craft.",
    "fork": "A utensil with prongs. Uses: food consumption.",
    "spoon": "A utensil for eating or serving food. Uses: food consumption.",
    "bottle": "A container for liquids, typically made of glass or plastic. Uses: beverage storage, transport.",
    "glass": "A transparent solid material. Uses: windows, drinking vessels, lenses.",
    "cup": "A small container for drinking. Uses: food consumption.",
    "plate": "A flat dish for serving food. Uses: food presentation.",
    "forklift": "A powered industrial truck. Uses: material handling in warehouses.",
    "shopping cart": "A cart for carrying groceries. Uses: retail shopping.",
    "toilet": "A fixture for human waste. Uses: sanitation.",
    "television": "An electronic device for viewing broadcasts. Uses: entertainment, news.",
    "computer": "An electronic device for processing data. Uses: communication, work.",
    "cell phone": "A portable device for communication. Uses: calls, messaging, internet access.",
    "camera": "A device for capturing images. Uses: photography, surveillance.",
    "backpack": "A bag carried on the back. Uses: carrying personal items.",
    "suitcase": "A case for carrying clothing. Uses: travel.",
    "umbrella": "A device for protection against rain or sun. Uses: weather protection.",
    "wallet": "A small case for holding money and cards. Uses: personal item storage.",
    "watch": "A device for measuring time. Uses: timekeeping.",
    "guitar": "A musical instrument. Uses: entertainment, music.",
    "drum": "A percussion instrument. Uses: music, rhythm.",
    "piano": "A musical instrument with keys. Uses: music, entertainment.",
    "microphone": "A device for capturing sound. Uses: audio recording, communication.",
    "speaker": "A device for outputting sound. Uses: music playback, public speaking.",
    "headphones": "A device for listening to audio. Uses: personal audio enjoyment.",
    "printer": "A device for producing paper documents. Uses: office work, document management.",
    "scissors": "A tool for cutting. Uses: crafting, office work.",
    "glue": "An adhesive substance. Uses: crafting, repairs.",
    "paintbrush": "A tool for applying paint. Uses: art, decoration.",
    "camera tripod": "A stand for stabilizing a camera. Uses: photography, videography.",
    "laptop": "A portable computer. Uses: personal and professional work.",
    "tablet": "A portable touch-screen computer. Uses: browsing, entertainment.",
    "router": "A device for networking. Uses: internet connectivity.",
    "flashlight": "A portable light source. Uses: illumination, emergency lighting.",
    "battery": "A device for storing electrical energy. Uses: powering devices.",
    "light bulb": "An electric light source. Uses: lighting.",
    "candle": "A wick encased in wax. Uses: lighting, ambiance.",
    "mirror": "A reflective surface. Uses: decoration, personal grooming.",
    "key": "A device for opening locks. Uses: security.",
    "lock": "A device for securing access. Uses: security.",
    "fence": "A barrier enclosing an area. Uses: security, property line demarcation.",
    "door": "A movable barrier for entry. Uses: access control.",
    "window": "An opening in a wall. Uses: light and ventilation.",
    "roof": "The top covering of a building. Uses: shelter, protection.",
    "wall": "A vertical structure. Uses: support, division.",
    "floor": "The bottom surface of a room. Uses: structural support.",
    "ceiling": "The overhead interior surface. Uses: protection from elements.",
    "table": "A piece of furniture with a flat surface. Uses: dining, work.",
    "chair": "A piece of furniture for sitting. Uses: seating.",
    "sofa": "A comfortable seat for multiple people. Uses: relaxation, seating.",
    "bed": "A piece of furniture for sleeping. Uses: rest, sleep.",
    "dresser": "A piece of furniture for storing clothes. Uses: storage.",
    "cabinet": "A cupboard with shelves. Uses: storage.",
    "shelf": "A flat horizontal surface. Uses: storage, display.",
    "rug": "A piece of fabric used on floors. Uses: decoration, comfort.",
    "curtain": "A piece of fabric hung to block light. Uses: privacy, decoration.",
    "plant": "A living organism that photosynthesizes. Uses: decoration, air purification.",
    "tv stand": "A piece of furniture for supporting a television. Uses: furniture organization.",
    "bookshelf": "A structure for storing books. Uses: organization, decoration.",
    "fireplace": "A structure for containing a fire. Uses: heating, ambiance.",
    "pool": "A large container of water for swimming. Uses: recreation, exercise.",
    "spectacles": "A pair of lenses set in a frame worn in front of the eyes. Uses: vision correction, protection from sun.",
    "sports ball": "A spherical object used in various sports. Uses: fitness, recreation, team sports, individual practice.",
    "tennis racket": "A bat with a long handle used to hit a tennis ball. Uses: sports, fitness.",
    "golf club": "A club used to hit a golf ball. Uses: sports, leisure.",
    "swimming goggles": "A pair of goggles for swimming. Uses: eye protection, visibility in water.",
    "yoga mat": "A mat used for yoga practice. Uses: exercise, fitness.",
    "skateboard": "A board mounted on wheels. Uses: recreation, sport.",
    "hockey stick": "A stick used to hit a puck in ice hockey. Uses: sports, recreation."


    # Add more classes and their descriptions as needed
}

# Initialize the model with user-defined parameters
@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
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
        vid_stride=1,  # video frame-rate stride
):
    # Set up the source for inference
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                detected_classes = {}
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    detected_classes[names[int(c)]] = n  # Count the detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Print out what was detected
                LOGGER.info(f'Detected: {detected_classes}')  # Log detected classes

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_crop:  # Save cropped prediction boxes
                        save_one_box(xyxy, imc, save_path=save_dir / 'crops' / names[int(cls)] / f'{p.stem}.jpg')

                    # Draw boxes and labels on the image
                    if hide_labels and hide_conf:
                        label = None
                    elif hide_labels:
                        label = f'{names[int(cls)]}'
                    elif hide_conf:
                        label = f'{names[int(cls)]} {conf:.2f}'
                    else:
                        label = f'{names[int(cls)]} {conf:.2f}'

                    annotator.box_label(xyxy, label, color=colors(cls, True))

                    # Print description and uses of detected object
                    if names[int(cls)] in class_descriptions:
                        description = class_descriptions[names[int(cls)]]
                        LOGGER.info(f'Detected: {names[int(cls)]} - {description}')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # video
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                        w, h = im0.shape[1], im0.shape[0]
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Release resources
    if save_txt or save_img:
        s = f"Results saved to {save_dir}"
    LOGGER.info(s)
    for i, v in enumerate(vid_writer):
        if isinstance(v, cv2.VideoWriter):
            v.release()  # release video writer

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640, 640], help='inference size (height, width)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', type=int, default=3, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    args = parser.parse_args()

    # Call the run function with arguments
    print_args(vars(args))  # This will display the arguments being used
    run(**vars(args))  # Run the detection
