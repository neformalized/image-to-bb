from hood import dataset, buffer, builder, image, label, log, report

class Handler():
    
    def __init__(self, dataset_train_path, dataset_validation_path, buffer_size, input_image, output_labels, loss_target, save_path):
        
        self.dataset_train_path = dataset_train_path
        self.dataset_train = dataset.Dataset(self.dataset_train_path)
        
        self.dataset_validation_path = dataset_validation_path
        self.dataset_validation = dataset.Dataset(self.dataset_validation_path)
        
        #
        
        self.buffer_size = buffer_size
        
        #
        
        self.input_image = input_image
        
        #
        
        self.output_labels = output_labels
        
        #
        
        self.model = builder.create(self.input_image, self.output_labels)
        
        #
        
        self.loss_target = loss_target
        
        #
        
        self.save_path = save_path
        
        #
        
        self.log = log.Log(self.save_path)
        
        #
        
        self.epoch = 1
        
        self.loss_train_current = 1.
        self.loss_train_last = 1.
        self.loss_train_record = 1.
        
        self.loss_validation_current = 1.
        self.loss_validation_last = 1.
        self.loss_validation_record = 1.
        
        #
        
        size_init = self.buffer_size if self.buffer_size > 0 else len(self.dataset_train.data)
        self.buffer_train = buffer.Dual(size_init, self.input_image, self.output_labels)
        
        if(self.buffer_size == 0):
            
            self.buffer_load(self.dataset_train.data, self.buffer_train)
        #
        
        #
        
        if(len(self.dataset_validation.data) > 0):
            
            self.buffer_validation = buffer.Dual(len(self.dataset_validation.data), self.input_image, self.output_labels)
            self.buffer_load(self.dataset_validation.data, self.buffer_validation)
        #
    #
    
    def start(self):
        
        while True:
            
            #
            
            self.fit()
            
            #
            
            self.evaluate()
            
            #
            
            self.checkpoint()
            
            #
            
            self.write_log()
            
            #
            
            if self.is_done(): break
            
            #
            
            self.updates()
        #
        
        if self.dataset_validation_path: report.visualise(self.model, self.buffer_validation, self.input_image)
        
        #
    #
    
    def fit(self):
        
        print("================")
        
        self.fit_full() if(self.buffer_size == 0) else self.fit_parts()
    #
    
    def evaluate(self):
        
        if not self.dataset_validation_path: return
        
        self.loss_validation_current = round(self.model.evaluate(self.buffer_validation.x, self.buffer_validation.y, batch_size = 1), 4)
    #
    
    def checkpoint(self):
        
        self.model.save(self.save_path + "model.keras")
        
        if(self.loss_validation_current < self.loss_validation_record):
            
            self.model.save(self.save_path + "model_evaluated.keras")
            
            print("record!")
        #
    #
    
    def write_log(self):
        
        self.log.line(self.epoch, self.loss_train_current, self.loss_validation_current)
    #
    
    def is_done(self):
        
        if(self.loss_validation_current <= self.loss_target): return True
        
        #
        
        return False
    #
    
    def updates(self):
        
        if(self.loss_validation_current > self.loss_validation_last):
            
            new_lr = self.model.optimizer.learning_rate * 0.76
            
            self.model.optimizer.learning_rate = new_lr
            
            print("new learning_rate")
            print(new_lr)
        #
        
        if(self.loss_train_current < self.loss_train_record): self.loss_train_record = self.loss_train_current
        if(self.loss_validation_current < self.loss_validation_record): self.loss_validation_record = self.loss_validation_current
        
        #
        
        self.loss_train_last = self.loss_train_current
        self.loss_validation_last = self.loss_validation_current
        
        #
        
        self.epoch += 1
    #
    
    def buffer_load(self, data, target_buffer):
        
        for i, item in enumerate(data):
            
            x = image.process(item[0], self.input_image)
            y = label.process(item[1])
            
            target_buffer.put(i, x, y)
        #
    #
    
    def fit_full(self):
        
        print("epoch#{}".format(self.epoch))
        
        #
        
        self.loss_train_current = round(self.model.fit(self.buffer_train.x, self.buffer_train.y, batch_size = 1, epochs = 1).history["loss"][0], 4)
        
        #
    #

    def fit_parts(self):
        
        print("epoch#{}".format(self.epoch))
        
        #
        
        train_len = len(self.dataset_train.data)
        
        #
        
        losses = []
        
        for start in range(0, train_len, self.buffer_size):
            
            end = min(start + self.buffer_size, train_len)
            
            #
            
            print("step {} from {}".format([start, end], train_len))
            
            #
            
            self.buffer_load(self.dataset_train.data[start: end], self.buffer_train)
            
            #
            
            losses.append(round(self.model.fit(self.buffer_train.x[0: end - start], self.buffer_train.y[0: end - start], batch_size = 1, epochs = 1).history["loss"][0], 4))
            
            #
        #
        
        self.loss_train_current = round(sum(losses)/len(losses), 4)
        
        #
    #
#