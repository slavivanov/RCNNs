import os
import cv2
import xml.etree.ElementTree as ET


def get_data(input_path, set_name='train', year='2007', use_difficult=False):
    all_imgs = []

    visualise = False

    data_paths = [os.path.join(input_path, s) for s in ['VOC' + year]]

    print('Parsing annotation files')

    for data_path in data_paths:
        annot_path = os.path.join(data_path, 'Annotations')
        imgs_path = os.path.join(data_path, 'JPEGImages')
        imgsets_path = os.path.join(
            data_path, 'ImageSets', 'Main', set_name + '.txt')

        set_files = []
        try:
            with open(imgsets_path) as f:
                for line in f:
                    set_files.append(line.strip())
        except Exception as e:
            print(e)

        set_images = [name + '.jpg' for name in set_files]
        annots = [os.path.join(annot_path, s + '.xml')
                  for s in set_files]
        idx = 0
        for annot in annots:
            try:
                idx += 1

                et = ET.parse(annot)
                element = et.getroot()
                
                element_objs = element.findall('object')
                element_filename = element.find('filename').text
                element_width = int(element.find('size').find('width').text)
                element_height = int(element.find('size').find('height').text)

                if len(element_objs) > 0:
                    annotation_data = {
                        'filepath': os.path.join(imgs_path, element_filename),
                        'filename': element_filename,
                        'width': element_width,
                        'height': element_height, 
                        'bboxes': [],
                        'imageset': set_name}
                    

                for element_obj in element_objs:
                    # Are we using difficult objects
                    if not use_difficult:
                        if int(element_obj.find('difficult').text) == 1:
                            continue
                            
                    class_name = element_obj.find('name').text                        

                    obj_bbox = element_obj.find('bndbox')
                    x1 = int(round(float(obj_bbox.find('xmin').text)))
                    y1 = int(round(float(obj_bbox.find('ymin').text)))
                    x2 = int(round(float(obj_bbox.find('xmax').text)))
                    y2 = int(round(float(obj_bbox.find('ymax').text)))
                    annotation_data['bboxes'].append(
                        {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})

                all_imgs.append(annotation_data)

                if visualise:
                    img = cv2.imread(annotation_data['filepath'])
                    for bbox in annotation_data['bboxes']:
                        cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
                            'x2'], bbox['y2']), (0, 0, 255))
                    cv2.imshow('img', img)
                    cv2.waitKey(0)

            except Exception as e:
                print(e)
                continue
    return all_imgs