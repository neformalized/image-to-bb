# image-to-bb

Simple, stable and rapidly solution to learn image-to-bounding_box models

https://colab.research.google.com/drive/197-yMBIJNYKqiZvkiz-WuZR2u5u5-XUr?usp=sharing (Try it yourself simply)

#

Edit init file for pipeline specific configuration

/source/init.py

#

Learning Loop is easy to modify

/source/hood/handler.py - strart()


while True:

  self.fit() # fit full train dataset
  
  self.evaluate() # evaluate validation dataset
  
  self.checkpoint() # save weights
  
  self.write_log() # write epoch log
  
  if self.is_done(): break # break condition state
  
  self.updates() # update variables, model, etc
  

#

Pipeline Struct is easy to modify as well


*hood/buffer.py # transformed x,y data holder class

*hood/builder.py # keras model struct

*hood/dataset.py # dataset holder class

*hood/image.py # model input encoder

*hood/label.py # model output encoder

*hood/log.py # log class


*hood/handler.py # main pipeline class

#

Feel free to contact me if you have any question!
https://www.linkedin.com/in/sergey-syschenko-027b01318/
