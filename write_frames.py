import cv2
import glob

#dataset path
path = 'C:/Users/igorc/Desktop/Maxtrack/MaxTrack_NIR/*.avi'

#training coefficient
coefficient = 0.9

#storing number of desired frames
number_of_desired_frames = 6

if __name__ == "__main__":
   
   list_of_videos = glob.glob(path)
   list_of_videos = list_of_videos[1:]
   #print(list_of_videos)

   image_counter = 0
    
   for video in list_of_videos: 
       vidcap = cv2.VideoCapture(video)
       number_of_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
       
       #computing the sampling rate on the video 
       step_factor = int(number_of_frames/number_of_desired_frames)
        
       success,image = vidcap.read()
       
       frame_counter = 0
       while success:
          success,image = vidcap.read()
          
          if success:
              #resizing frame  
              image = cv2.resize(image,(192,192))

              #writing on test or train folders
              writing_path = ''
              if frame_counter < coefficient*number_of_frames:
                  writing_path = 'C:/Users/igorc/Desktop/MaxTrack_Basic/trainB/'
              else:
                  writing_path = 'C:/Users/igorc/Desktop/MaxTrack_Basic/testB/' 

              cv2.imwrite(writing_path + str(image_counter) + '.png', image)

              #adjusting video pointer
              vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
              frame_counter = frame_counter + step_factor

              #adjusting outside counter  
              image_counter = image_counter + 1  
  