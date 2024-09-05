import cv2

def process(image_path, target_size):
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = min(target_w / w, target_h / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h))
    
    padded_image = cv2.copyMakeBorder(
        
        resized_image,
        top=(target_h - new_h) // 2,
        bottom=(target_h - new_h) // 2,
        left=(target_w - new_w) // 2,
        right=(target_w - new_w) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    
    processed = cv2.resize(padded_image, (target_w, target_h))
    processed = processed/255
    
    return processed
#
