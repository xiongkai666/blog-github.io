import numpy as np
 
def ComputeIOU(boxA, boxB):
    ## 计算相交框的坐标
    x1 = np.max([boxA[0], boxB[0]])
    x2 = np.min([boxA[2], boxB[2]])
    y1 = np.max([boxA[1], boxB[1]])
    y2 = np.min([boxA[3], boxB[3]])
    
    width = np.max([0, x2 - x1 + 1])
    height = np.max([0, y2 - y1 + 1])
    inter_area = width * height
    
    # 计算两个框的面积
    area_A = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    area_B = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # 计算并区域面积
    union_area = area_A + area_B - inter_area
    
    # 计算IOU
    iou = inter_area / union_area
    return iou

if __name__ == "__main__":
    boxA = [1,1,3,3] # A : [x1, y1, x2, y2]
    boxB = [2,2,4,4] 
    IOU = ComputeIOU(boxA, boxB)
    print(f"IOU: {IOU:.4f}") 