''' 
TASK: 
Source prodcuing new images (1024x768x3 channels) every 50ms.  
Propose an architecture producer-consumer 
    *producer (thread) have a source of images. Every 50ms get images from source and put in the queue A
    *consumer takes the image available in queue A, reduces its size twice, apply median filter with kernel 5x5
     and put new image  in the queue B
'main' function processe 100 images and stops running. All 100 images should be stored in png format in 
'processed' direcotry.  
'''


'''
SOLUTION:
Proposed architecture:  1 producer and many threads. Seems
that there is no need to run multiple threads of the producer when the source produces
slower than the manufacturer processes. This is how I understand the statment that the manufacturer downloads
data from source every 50ms.

Since Python processes threads concurrently but not in parallel there will be no  gain in time from starting multiple threads
We also won't see any gain in time of the manufacturer's process 
and the consumer will be slower than the ratepace  at which the source produces new images.
'''

import threading
import queue
import source
import time
import cv2 
import os 

end_of_cons=threading.Event()

def make_dir(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

def write_images(path,queue):
    make_dir(path)
    for name in range(queue.qsize()):
        image=queue.get()
        cv2.imwrite(path+str(name)+'.png',image)

def transform_image(image):
    image = cv2.resize(image, (int(image.shape[0]/2),int(image.shape[1]/2)), interpolation = cv2.INTER_LINEAR)
    image = cv2.medianBlur(image,5)
    return image

def producer(source, queue_producer,images_count):
    while images_count>=0:
        queue_producer.put(source.get_data())
        images_count-=1
        time.sleep(0.05)
    
def consumer(queue_producer,queue_consumer,images_count):
    
    while not end_of_cons.is_set():
            try:   
             image=queue_producer.get_nowait()
            except: pass
            else :
                image=transform_image(image)
                try :
                    queue_consumer.put_nowait(image)
                except:
                    end_of_cons.set()

def main():

    path = './processed/'
    rows = 786
    cols = 1024
    channels = 3

    images_count = 100
    consumer_threads_count = 3

    source_of_image=source.Source((rows,cols,channels))

    queue_producer=queue.Queue(maxsize=images_count)
    queue_consumer=queue.Queue(maxsize=images_count)


    prod = threading.Thread(target=producer,args=(source_of_image,
                                                    queue_producer,
                                                    images_count))

    cons =[ threading.Thread(target=consumer,args=(queue_producer,
                                                    queue_consumer,
                                                    images_count,))
            for _ in range(consumer_threads_count) ]

    t=time.time()
    
    prod.start()
    for c in cons: c.start()

    prod.join()
    for c in cons: c.join()

    write_images(path,queue_consumer)

    print('Czas dla',consumer_threads_count,'watkow', time.time()-t)
    

if __name__=='__main__':main()


