import cv2
import json
import matplotlib.pyplot as plt


# convert native annotations to COCO Format
def convert_to_coco(ann_json, image_file_path, img_id, ann_id):
    # read image to get width and height using cv2
    image = cv2.imread(image_file_path)
    img_height, img_width = image.shape[:2]
    
    if ann_json['type'] == 'rectangle':
        left = int(ann_json['coords']['left'] * img_width)
        top = int(ann_json['coords']['top'] * img_height)
        width = int(ann_json['coords']['width'] * img_width)
        height = int(ann_json['coords']['height'] * img_height)
        coco_ann = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [left, top, width, height],
            "area": width * height,
            "iscrowd": 0
        }
    elif ann_json['type'] == 'polygon':
        points = ann_json['coords']['points']
        segmentation = []
        for point in points:
            x = int(point['x'] * img_width)
            y = int(point['y'] * img_height)
            segmentation.extend([x, y])
        coco_ann = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": [segmentation],
            "area": 0,  # Area calculation can be added if needed
            "iscrowd": 0
        }
    return coco_ann


# visualize COCO annotations
def visualize_coco_annotations(image_file_path, coco_json):
    # read image
    image = cv2.imread(image_file_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for ann in coco_json['annotations']:
        if 'bbox' in ann:
            left, top, width, height = ann['bbox']
            rect = plt.Rectangle((left, top), width, height, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            # also label the annotations with their ids
            plt.text(left, top, 'lesion-'+str(ann['id']), color='yellow', fontsize=12, backgroundcolor='black')
            
        elif 'segmentation' in ann:
            segmentation = ann['segmentation'][0]
            polygon_xy = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
            polygon = plt.Polygon(polygon_xy, linewidth=5, edgecolor='b', facecolor='none')
            plt.gca().add_patch(polygon)
            # also label the annotations with their ids
            plt.text(polygon_xy[0][0], polygon_xy[0][1], 'lesion-'+str(ann['id']), color='yellow', fontsize=12, backgroundcolor='black')
    plt.axis('off')
    plt.show()
    
if __name__ == '__main__':
    # Example usage    
    image_file_path = 'sample_image/Original Image.jpeg'
    
    # read sample annotation jsons
    with open('sample_native_annotations/sample_annotations_v2.json', 'r') as f:
        ann_json = f.read()
    ann_json = json.loads(ann_json)
    img_id = 1
    ann_id = 1
    
    for each_ann in ann_json:
        coco_ann = convert_to_coco(each_ann, image_file_path, img_id, ann_id)
        print(coco_ann)
        ann_id += 1
        
    visualize_coco_annotations(image_file_path, {
        "annotations": [convert_to_coco(each_ann, image_file_path, img_id, idx+1) for idx, each_ann in enumerate(ann_json)]
    })