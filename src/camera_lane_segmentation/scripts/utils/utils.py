import datetime
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
import re
import glob
import random
import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def git_describe(path=Path(__file__).parent):
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError:
        return ''

def date_modified(path=__file__):
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'

def select_device(device='', batch_size=None):
    s = f'YOLOPv2 🚀 {git_describe() or date_modified()} torch {torch.__version__} '
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"
    else:
        s += 'CPU\n'

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)
    return torch.device('cuda:0' if cuda else 'cpu')

def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, [0, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, [0, 255, 255], -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def lineAccuracy(self):
        Acc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1) + 1e-12)
        return Acc[1]

    def classPixelAccuracy(self):
        classAcc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + 1e-12)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + \
                np.sum(self.confusionMatrix, axis=0) - \
                np.diag(self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        mIoU = np.nanmean(IoU)
        return mIoU

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + \
                np.sum(self.confusionMatrix, axis=0) - \
                np.diag(self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        return IoU[1]

    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
            np.sum(self.confusionMatrix, axis=1) +
            np.sum(self.confusionMatrix, axis=0) -
            np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def split_for_trace_model(pred=None, anchor_grid=None):
    z = []
    st = [8, 16, 32]
    for i in range(3):
        bs, _, ny, nx = pred[i].shape
        pred[i] = pred[i].view(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        y = pred[i].sigmoid()
        gr = _make_grid(nx, ny).to(pred[i].device)
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + gr) * st[i]
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]
        z.append(y.view(bs, -1, 85))
    pred = torch.cat(z, 1)
    return pred

def show_seg_result(img, result, palette=None, is_demo=False):
    if palette is None:
        palette = np.random.randint(0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3
    assert palette.shape[1] == 3

    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
        color_area[result[0] == 1] = [0, 255, 0]
        color_area[result[1] == 1] = [255, 0, 0]
        color_seg = color_area

    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5

def increment_path(path, exist_ok=True, sep=''):
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{sep}{n}"

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    boxes[:, 0].clamp_(0, img_shape[1])
    boxes[:, 1].clamp_(0, img_shape[0])
    boxes[:, 2].clamp_(0, img_shape[1])
    boxes[:, 3].clamp_(0, img_shape[0])

def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45,
                        classes=None, agnostic=False, multi_label=False,
                        labels=()):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    min_wh, max_wh = 2, 4096
    max_det = 300
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break
    return output

def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)

class LoadCamera:
    def __init__(self, source, img_size=640, stride=32):
        self.source = source
        self.img_size = img_size
        self.stride = stride
        try:
            cam_index = int(self.source)
            self.cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
            # self.cap = cv2.VideoCapture(cam_index)
        except ValueError:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
            # self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise Exception(f"카메라를 열 수 없습니다. ID/Dev: {self.source}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # AUTOFOCUS 끄기

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Camera opened: {self.width}x{self.height} @ {self.fps} FPS")

        self.mode = 'stream'
        self.frame = 0

    def __iter__(self):
        return self

    def __next__(self):
        ret, img0 = self.cap.read()
        if not ret:
            raise StopIteration
        self.frame += 1

        # YOLO 입력용 letterbox
        img = letterbox(img0, (self.img_size, self.img_size), stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return '', img, img0, self.cap

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

class LoadImages:
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))
        elif os.path.isfile(p):
            files = [p]
        else:
            raise Exception(f'ERROR: {p} does not exist')

        img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
        vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.count = 0  # <-- 객체 생성 시 카운터를 0으로 초기화
        self.cap = None

        if any(videos):
            self.new_video(videos[0])
        else:
            self.cap = None

        assert self.nf > 0, f'No images or videos found in {p}. '

    def __iter__(self):
        self.count = 0  # <-- 매번 순회(iter) 시작 시 0으로 초기화
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()
            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')
        else:
            self.mode = 'image'
            img0 = cv2.imread(path)
            assert img0 is not None, 'Image Not Found ' + path
            self.count += 1

        # 1280x720로 강제 리사이즈
        img0 = cv2.resize(img0, (1280, 720))

        # YOLO 입력용 letterbox
        img = letterbox(img0, (self.img_size, self.img_size), stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def driving_area_mask(seg=None):
    da_predict = seg[:, :, 12:372, :]
    da_seg_mask = F.interpolate(da_predict, scale_factor=2, mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
    return da_seg_mask

def lane_line_mask(ll=None, threshold=0.5, method='otsu'):
    """
    차선 세그멘테이션 결과로부터 이진 마스크를 생성합니다.
    
    Parameters:
        ll: torch.Tensor
            네트워크의 차선 세그멘테이션 출력 (배치, 채널, 높이, 너비)
        threshold: float
            고정 임계값 방식 사용 시 적용할 임계값 (0~1 사이)
        method: str, 선택 사항
            'fixed'   : 고정 임계값 방식 (기본값)
            'otsu'    : Otsu thresholding 방식
             
    Returns:
        binary_mask: numpy.ndarray
            0과 255로 구성된 이진 마스크
    """
    # (1) 관심 영역(crop) 선택 및 해상도 보정
    ll_predict = ll[:, :, 12:372, :]  # 원래 코드와 동일
    ll_seg_map = F.interpolate(ll_predict, scale_factor=2, mode='bilinear')
    ll_seg_map = ll_seg_map.squeeze(1)  # shape: (B, H, W)
    
    # 배치 사이즈가 1이라고 가정하고 첫 번째 결과 사용
    ll_seg_map = ll_seg_map[0]  # shape: (H, W)
    # tensor → numpy (GPU tensor인 경우 .cpu() 필요)
    ll_seg_map = ll_seg_map.cpu().numpy()
    
    # guided filter 적용: 먼저 0~255 범위의 8비트 이미지로 변환
    ll_seg_map_8u = (ll_seg_map * 255).astype(np.uint8)
    # guided filter 파라미터: radius와 eps는 데이터에 따라 튜닝 필요
    guided = cv2.ximgproc.guidedFilter(guide=ll_seg_map_8u, src=ll_seg_map_8u, radius=4, eps=1e-1)
    
    if method == 'fixed':
        binary_mask = (guided > threshold * 255).astype(np.uint8) * 255
    elif method == 'otsu':
        ret, binary_mask = cv2.threshold(guided, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError("Invalid method for binarization. Choose 'fixed' or 'otsu'.")
    
    return binary_mask


def apply_clahe(image):
    """
    CLAHE를 사용하여 이미지의 대비를 향상시킵니다.
    
    Parameters:
        image (numpy.ndarray): 입력 이미지. 컬러(BGR) 또는 그레이스케일.
        
    Returns:
        enhanced_image (numpy.ndarray): 대비가 향상된 이미지.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # 컬러 이미지인 경우 YUV 색 공간으로 변환 후 Y 채널에 CLAHE 적용
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        enhanced_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        # 그레이스케일 이미지인 경우 직접 CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)
    return enhanced_image
