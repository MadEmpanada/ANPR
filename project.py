#Imports
import argparse
import supervision as sv
import cv2
from ultralytics import YOLO
from pathlib import Path

def main():
    parameters = get_car_detections(arguments().source_video_path)
    output_detection(*parameters)

#Create a parser for obtaining video file path
def arguments():
    parser = argparse.ArgumentParser(
                prog="Automatic Number Plate Recognition",
                description="Analyses number plates from cars using YOLOv10 as the object detection model."
        )
    parser.add_argument(
            "source_video_path", 
            help="Path to source video file",
            type=str
        )
    return parser.parse_args()

def get_car_detections(video_path) -> list:
    
    images = []
    
    #Get frames and video parameters
    frames = sv.get_video_frames_generator(video_path)
    video_info = sv.VideoInfo.from_video_path(video_path)

    #Define bounding box and label parameters
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    text_font = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    label = sv.LabelAnnotator(text_scale=text_font, text_thickness=thickness)

    #Load models for car detection and license plate detection
    car_model = YOLO("yolov10n.pt")

    for frame in frames:
        
        #Find the car detections on the current frame
        car_results = car_model(frame)[0]
        car_detections = sv.Detections.from_ultralytics(car_results)
        
        written_frame = frame.copy()
        
        #Filter the model detections exclusively for cars
        for class_id in car_detections.class_id:
            if class_id == 2:
                
                #Create a copy of the current frame to writte labels and draw boxes on it based on the detections made by the model
                #Detect specifically cars whith their corresponding class_id
                written_frame = bounding_box_annotator.annotate(scene=written_frame, detections=car_detections[car_detections.class_id == class_id])
                written_frame = label.annotate(scene=written_frame, detections= car_detections[car_detections.class_id == class_id])
            
        #Resizing output to fit the window
        h, w = written_frame.shape[:2] #unpacks only 2 values which correspond to the height and the width of the frame
        scale = 25
        width = int(w * scale/100)
        height = int(h * scale/100)
        dim = (width, height)
        display_frame = cv2.resize(written_frame, dim, interpolation=cv2.INTER_AREA)
        
        #Append the frame to display in the output video into the list 
        images.append(display_frame)
        
        #Show results on live
        cv2.imshow("display_frame", display_frame)
        if cv2.waitKey(1) == ord("x"):
            break
        
    return [images, width, height]

def output_detection(images, width, height) -> bool:
    
    video_name = "car_detections.mp4"
    video = cv2.VideoWriter(video_name, 0, 30, (width, height))
    for image in images:
        video.write(image)
    
    if Path("car_detections.mp4").exists():
        return True
    else:
        return False
        
if __name__ == "__main__":
    main()
