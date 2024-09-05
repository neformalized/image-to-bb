from hood import handler

if __name__ == "__main__":
    
    #
    
    dataset_path_train = ["/content/dataset/train.txt", "/content/dataset/train/"]
    dataset_path_validation = ["/content/dataset/validation.txt", "/content/dataset/validation/"]
    #dataset_path_validation = False
    
    #
    
    buffer_size = 0 # 0 whole dataset without reloads
    
    #
    
    input_image = (139, 132)
    
    #
    
    output_labels = 4
    
    #
    
    
    loss_target = 0.05
    
    #
    
    save_path = "/content/"
    
    #
    
    handler = handler.Handler(dataset_path_train, dataset_path_validation, buffer_size, input_image, output_labels, loss_target, save_path)
    handler.start()
    
    #
#