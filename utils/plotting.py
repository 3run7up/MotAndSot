import numpy as np
import cv2
from ultralytics import YOLO

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython import display
import ultralytics.utils.plotting as plotting

from collections import defaultdict
import time

COLOURS = [
        'aqua', 'aquamarine', 'blue', 'crimson', 'brown', 'burlywood', 'yellow',
        'orange', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'blueviolet', 'cyan',
        'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
        'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
        'darkseagreen', 'darkslateblue', 'chartreuse', 'red', 'darkturquoise',
        'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
        'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
        'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
        'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
        'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
        'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
        'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
        'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
        'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
        'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
        'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
        'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
        'purple', 'rebeccapurple', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
        'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
        'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
        'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellowgreen'
    ]

def cv2_plot_tracks(frame, result, track_history: defaultdict, colours: list):
    annotated_frame = frame.copy()
    boxes = result.boxes

    if (boxes.is_track):
        boxes_xywh = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        
        annotator = plotting.Annotator(annotated_frame, line_width=6, font_size=1)
        for box, track_id in zip(boxes_xywh, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  
            if len(track) > 15:  
                track.pop(0)
            xyxy = (x - w / 2, y - h/2, x + w / 2, y + h/2)

            color = colours[track_id % len(colours)]
            annotator.box_label(xyxy, color=color) #f"id {track_id}"
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=7)
            cv2.putText(annotated_frame,
                        f"id {track_id}", 
                        (int(x - w // 2), int(y - h //2)),
                        0,
                        1,
                        color,
                        thickness=6,
                        lineType=cv2.LINE_AA)

    return annotated_frame

def tracker_inference_video(model_name, input_video_path, output_video_path, config, max_img = None):
    def generate_colour():
        r, g, b = np.random.randint(1, 255, (3))
        return (int(r), int(g), int(b))
    CV2_COLOURS = [generate_colour() for i in range(100)]

    model = YOLO(model_name)
    track_history = defaultdict(lambda: [])
    
    if(max_img is None):
        max_img = np.inf

    processed_img = 0

    input_cap = cv2.VideoCapture(input_video_path)
    width  = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
    height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fps = input_cap.get(cv2.CAP_PROP_FPS)
    out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('F','M','P','4'), fps, (width, height))


    while input_cap.isOpened() and processed_img < max_img:
        success, frame = input_cap.read()

        if success:
            results = model.track(frame, persist=True, **config)

            img_to_save = cv2_plot_tracks(frame, results[0], track_history, CV2_COLOURS)
            # img_show = cv2.cvtColor(img_to_save, cv2.COLOR_BGR2RGB)
            # fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            # ax.axis('off')
            # ax.imshow(img_show)
            out_video.write(img_to_save)
            
            processed_img += 1
        else:
            break

    input_cap.release()
    out_video.release()

def plt_show_tracker(video_path, model, max_count_processed_img, model_config, time_sleep: float = None):
  track_history = defaultdict(lambda: [])
  processed_img = 0
  fig, ax = plt.subplots(1, 1, figsize=(14, 14))

  cap = cv2.VideoCapture(video_path)

  while cap.isOpened() and processed_img < max_count_processed_img:
      success, frame = cap.read()

      if success:
          results = model.track(frame, persist=True, **model_config)

          ax.clear()
          display.clear_output(wait=True)
          plot_tracks_result(frame, results[0], track_history, COLOURS, ax = ax)
          display.display(fig)
          if (time_sleep):
              time.sleep(time_sleep)

          processed_img += 1
      else:
          break

  fig.clear()
  cap.release()


def plot_track_history(ax, track_history, color):
    x = []
    y = []
    for track in track_history:
        x.append(track[0])
        y.append(track[1])
    ax.plot(x, y, color=color, marker="o", markersize=4, linewidth=2)

def add_obj_bbox_to_plot(ax, box, color, display_id = None):
    x, y, w, h = box
    ax.add_patch(patches.Rectangle(
            (x - w // 2, y - h // 2),
            w,
            h,
            fill=False,
            linewidth=3,
            edgecolor=color,
        ))
    
    if not display_id is None:
        ax.text(x - w // 2, y - h // 2, f'id: {display_id}', fontsize=10, color=color)
     


def plot_tracks_result(frame, result, track_history: defaultdict, colours: list = None, ax = None):
    if colours is None:
        colours = COLOURS
    if (ax is None):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    
    boxes = result.boxes.xywh.cpu()
    track_ids = result.boxes.id.int().cpu().tolist()
    img_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.axis('off')
    ax.imshow(img_show)
    

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  
        if len(track) > 15:  
            track.pop(0)
            
        color = colours[track_id % len(colours)]
        add_obj_bbox_to_plot(ax, box, color, track_id)
        plot_track_history(ax, track, color)

def plot_detection_result(result, ax = None):
    annotated_frame = result.plot()
    img_show = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.axis('off')
    ax.imshow(img_show)