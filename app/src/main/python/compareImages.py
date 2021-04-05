import numpy as np
import cv2
from PIL import Image
from deepface import DeepFace
import base64
import io

def compareImages(registeredImg, compareWithImg):
    decodedRegImg= base64.b64decode(registeredImg)
    decodedCompareImg= base64.b64decode(compareWithImg)
    
    npRegImg= np.fromstring(decodeRegImg, np.uint8)
    npCompImg= np.fromstring(decodedCompareImg, np.uint8)
    
    regImg= cv2.imdecode(npRegImg, cv2.IMREAD_UNCHANGED)
    compImg= cv2.imdecode(npCompImg, cv2.IMREAD_UNCHANGED)
    
    result  = DeepFace.verify(regImg, compImg)
    
    return result["verified"]