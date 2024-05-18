import xmltodict
import cv2
import numpy as np


def load_anns_file(path):
    with open(path) as fd:
        anns_file = xmltodict.parse(fd.read())
    return anns_file


def get_anns_dict(anns_file, image_id):
    anns_dict = {}
    img_dict = anns_file["annotations"]["image"][image_id]
    if "polyline" not in img_dict:
        return {}
    if type(img_dict["polyline"]) is not list:
        img_dict["polyline"] = [img_dict["polyline"]]
    for pline in img_dict["polyline"]:
        pts = np.array([pt.split(",") for pt in pline["@points"].split(";")], dtype=float)
        if pline["@label"] in anns_dict:
            # anns_dict[pline["@label"]] = np.concatenate((anns_dict[pline["@label"]], pts))
            anns_dict[pline["@label"]].append(list(pts))
        else:
            anns_dict[pline["@label"]] = []
            anns_dict[pline["@label"]].append(list(pts))
    return anns_dict


def get_n_stitches(anns_dict):
    try:
        return len(anns_dict["Stitch"])
    except KeyError:
        return 0


def get_mask(image, anns_dict):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for key, coords_list in anns_dict.items():
        if key == "Stitch":
            color = 125
        else:
            color = 255
        for pts in coords_list:
            for i in range(len(pts) - 1):
                cv2.line(mask, tuple(pts[i].astype(np.uint8)), tuple(pts[i + 1].astype(np.uint8)), color, 1)
    """ Second option to draw lines
    draw_points = (np.asarray([pts[:,0], pts[:,1]]).T).astype(np.int32)
    cv2.polylines(mask3, [draw_points], False, (255,255,255))
    """
    return mask
