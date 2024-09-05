import numpy

class Dual:

    def __init__(self, size, shape_x, shape_y):
        
        self.x = numpy.zeros((size, shape_x[0], shape_x[1], 3), dtype = numpy.float16)
        self.y = numpy.zeros((size, shape_y), dtype = numpy.float16)
    #
    
    def put(self, index, x, y):
        
        self.x[index,:,:,:] = x
        self.y[index,:] = y
    #
#