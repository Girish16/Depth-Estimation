
# import libraries
import torch
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
'https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD'
# download ssd model
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

# apply model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ssd_model.to(device)
ssd_model.eval()

"This work is my own implementation"
# convert image to tensor
def img_loader(img_path):
  image = Image.open(img_path).resize((300,300)).convert('RGB')
  loader = transforms.Compose([transforms.ToTensor()])
  image = loader(image).unsqueeze(0)
  return image.to(device, torch.float)

# for annotation of images
def annotate_img(img_path, x, y, w, h, count):
  img = Image.open(img_path).resize((900,900)).convert('RGB')
  img.save('/drive/My Drive/test_dataset/images/img'+ str(count[0])+ '.jpg')
  with open('/drive/My Drive/test_dataset/annotations/img'+ str(count[0])+ '.txt', 'a') as f:
    f.write('Player'+ str(count[1])+ ' '+ str(x*3)+ ' '+ str(y*3)+ ' '+ str(w*3)+ ' '+ str(h*3)+ '\n')

# show image
def show_img(img_path, bboxes, classes, confidences, count):
  fig, ax = plt.subplots(1)
  image = Image.open(img_path).resize((300,300)).convert('RGB')
  ax.imshow(image)
  for idx in range(len(bboxes)):
    left, bot, right, top = bboxes[idx]
    x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, "{} {:.0f}%".format(classes[idx] - 1, confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
  plt.savefig('/drive/My Drive/Rv1/img'+ str(count)+ '.jpg')
  plt.show()

def read_annotations(dir, count):
  with open(dir+ '/img'+ str(count[0])+ '.txt', 'r') as f:
    for i in range(count[1]):
      f.readline()
    data = f.readline().split()
  return data


def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

# calculate matrix for testing
def calc_player_detection_metrics(x, y, w, h, count):
  d = read_annotations('/drive/My Drive/test_dataset/annotations', count)
  xa, ya, wa, ha = float(d[1]), float(d[2]), float(d[3]), float(d[4])
  iou = get_iou([xa,ya,xa+wa,ya+ha], [x*3, y*3, (x+w)*3, (y+h)*3])
  if iou > 0.5:
    pred = 'tp'
  elif iou == 0:
    pred = 'fn'
  else:
    pred = 'fp'
  return pred

# evaluate single images
def eval_img(tensor):
  with torch.no_grad():
    detections_batch = ssd_model(tensor)
  results_per_input = utils.decode_results(detections_batch)
  best_results_per_input = [utils.pick_best(results, 0.25) for results in results_per_input]
  return best_results_per_input

# crop and saving player images
def crop_and_save(img_path, x, y, w, h, count):
  img = Image.open(img_path).resize((900,900))
  cropped_img = img.crop((x*3, y*3, (x+w)*3, (y+h)*3))
  cropped_img.save('/drive/My Drive/test_dataset/players/img'+ str(count[0])+ '/player'+ str(count[1])+ '.jpg')
  print(cropped_img)

# for evaluating model
def evaluate_model(n, root_dir):
  tp, fp, fn = 0,0,0
  for i in range(1,n):
    img_name = 'IMG_9851_frame_'+ str(i).rjust(6,'0')+ '.JPG'
    img_path = root_dir+ '/'+ img_name
    img_tensor = img_loader(img_path)
    best_results = eval_img(img_tensor)
    for image_idx in range(len(best_results)):
      bboxes, classes, confidences = best_results[image_idx]

      show_img(img_path, bboxes, classes, confidences, i)

      for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
        pred = calc_player_detection_metrics(x, y, w, h, (i, idx))
        if pred == 'tp':
          tp += 1
        elif pred == 'fp':
          fp += 1
        else: fn += 1

    if i% 100 == 0:
      print(str(i)+ ' evaluated')

  print_results(tp, fp, fn)

def print_results(tp, fp, fn):
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  print('Precison: '+ str(precision))
  print('Recall: '+ str(recall))
  print('F-score: '+ str(2*(precision*recall)/(precision + recall)))
  print('Detection Speed: '+ str(17))