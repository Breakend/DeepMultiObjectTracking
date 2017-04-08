#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from scipy.io import loadmat
from collections import defaultdict
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

IM_WIDTH = 640
IM_HEIGHT = 480
IM_DEPTH = 3

# Hacky helper function
def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

all_obj = 0
data = defaultdict(dict)
for dname in sorted(glob.glob('data/annotations/set*')):
    set_name = os.path.basename(dname)
    data[set_name] = defaultdict(dict)
    for anno_fn in sorted(glob.glob('{}/*.vbb'.format(dname))):
        vbb = loadmat(anno_fn)
        nFrame = int(vbb['A'][0][0][0][0][0])
        objLists = vbb['A'][0][0][1][0]
        maxObj = int(vbb['A'][0][0][2][0][0])
        objInit = vbb['A'][0][0][3][0]
        objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
        objStr = vbb['A'][0][0][5][0]
        objEnd = vbb['A'][0][0][6][0]
        objHide = vbb['A'][0][0][7][0]
        altered = int(vbb['A'][0][0][8][0][0])
        log = vbb['A'][0][0][9][0]
        logLen = int(vbb['A'][0][0][10][0][0])

        video_name = os.path.splitext(os.path.basename(anno_fn))[0]
        data[set_name][video_name]['nFrame'] = nFrame
        data[set_name][video_name]['maxObj'] = maxObj
        data[set_name][video_name]['log'] = log.tolist()
        data[set_name][video_name]['logLen'] = logLen
        data[set_name][video_name]['altered'] = altered
        data[set_name][video_name]['frames'] = defaultdict(list)

        rel_path = anno_fn[:-4]
        n_obj = 0
        for frame_id, obj in enumerate(objLists):
            print frame_id
            # Prepare xml base structure
            root = 0
            root = ET.Element('annotation')
            folder = ET.SubElement(root, 'folder')
            filename = ET.SubElement(root, 'filename')

            folder.text = set_name
            filename.text = set_name+'_'+video_name+'_'+str(frame_id)+'.jpg'

            size = ET.SubElement(root, 'size')
            width = ET.SubElement(size, 'width')
            width.text = str(IM_WIDTH)
            height = ET.SubElement(size, 'height')
            height.text = str(IM_HEIGHT)
            depth = ET.SubElement(size, 'depth')
            depth.text = str(IM_DEPTH)

            if len(obj) > 0:
                for id, pos, occl, lock, posv in zip(
                        obj['id'][0], obj['pos'][0], obj['occl'][0],
                        obj['lock'][0], obj['posv'][0]):
                    keys = obj.dtype.names
                    id = int(id[0][0]) - 1  # MATLAB is 1-origin
                    pos = pos[0].tolist()
                    occl = int(occl[0][0])
                    lock = int(lock[0][0])
                    posv = posv[0].tolist()

                    datum = dict(zip(keys, [id, pos, occl, lock, posv]))
                    datum['lbl'] = str(objLbl[datum['id']])
                    datum['str'] = int(objStr[datum['id']])
                    datum['end'] = int(objEnd[datum['id']])
                    datum['hide'] = int(objHide[datum['id']])
                    datum['init'] = int(objInit[datum['id']])
                    data[set_name][video_name][
                        'frames'][frame_id].append(datum)

                    obj_node = ET.Element('object')
                    name = ET.SubElement(obj_node, 'name')
                    name.text = 'person'
                    bndbox = ET.SubElement(obj_node, 'bndbox')

                    xmin = ET.SubElement(bndbox, 'xmin')
                    ymin = ET.SubElement(bndbox, 'ymin')
                    xmax = ET.SubElement(bndbox, 'xmax')
                    ymax = ET.SubElement(bndbox, 'ymax')

                    if sum(pos) > 0:
                        if occl == 0:
                            xmin.text = str(pos[0])
                            ymin.text = str(pos[1])
                            xmax.text = str(pos[0] + pos[2])
                            ymax.text = str(pos[1] + pos[3])
                        else:
                            xmin.text = str(posv[0])
                            ymin.text = str(posv[1])
                            xmax.text = str(posv[0] + posv[2])
                            ymax.text = str(posv[1] + posv[3])
                        root.append(obj_node)


                    n_obj += 1
            tree = prettify(root)
            if len(obj) > 0:
                with open('data/annotations/'+set_name+'_'+video_name+'_'+str(frame_id)+'.xml', 'wb') as xml_file:
                    xml_file.write(tree)

        print(dname, anno_fn, n_obj)
        all_obj += n_obj

print('Number of objects:', all_obj)
