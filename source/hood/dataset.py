import os

class Dataset:

    def __init__(self, dataset_path):
        
        #
        
        if not dataset_path:
            
            self.data = []
            return        
        #
        
        self.dataset_path = dataset_path
        
        with open(self.dataset_path[0], "r") as file:
            
            self.data = self.excract(file.readlines())
        #
	#
    
    def excract(self, lines):
        
        result = []
        
        for line in lines:
            
            tmp = line.split(":")
            
            x = self.dataset_path[1] + tmp[0]
            y = tmp[1].replace("\n", "")
            
            if not os.path.isfile(x): continue
            
            result.append([x, y])
            
            #
        #
        
        return result
    #
#