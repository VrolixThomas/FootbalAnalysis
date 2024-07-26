from ultralytics import YOLO

import supervision as sv
import pickle
import os
import cv2

from Persons import Person, Player, Referree, Keeper
from utils import get_center_of_bbox, get_bbox_width
    
    
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack() 

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            print(f"Processing frames {i} to {i+batch_size} from {len(frames)}")
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections
        
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if stub_path is not None:
            person_path = os.path.join(stub_path, "person_dict.pkl")
            ball_path = os.path.join(stub_path, "ball.pkl")
            detection_path = os.path.join(stub_path, "detections.pkl")
          
                
        if read_from_stub and stub_path is not None and os.path.exists(person_path):
            with open(person_path,'rb') as f:
                person_dict = pickle.load(f)
            with open(ball_path,'rb') as f:
                ball = pickle.load(f)
            return (person_dict, ball)
        
        if os.path.exists(detection_path):
            with open(detection_path,'rb') as f:
                detections = pickle.load(f)
        else:
            detections = self.detect_frames(frames)
            with open(detection_path,'wb') as f:
                pickle.dump(detections,f)
        

        
        ball = []
        person_dict = {}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format - just a different way of representation
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Track Objects - adds tracker
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            seen_track_ids = set()
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                seen_track_ids.add(track_id)
                
                current_person = person_dict.get(track_id)
                
                if current_person:
                    current_person.add_bbox(bbox)
                else:
                    if cls_id == cls_names_inv['player']:
                        person_dict[track_id] = Player(track_id, frame_num)
                    elif cls_id == cls_names_inv['referee']:
                        person_dict[track_id] = Referree(track_id, frame_num)
                    else:
                        person_dict[track_id] = Keeper(track_id, frame_num)
        
                    person_dict[track_id].add_bbox(bbox)
            for track_id in list(person_dict.keys()):
                if track_id not in seen_track_ids:
                    person_dict[track_id].add_bbox(None)
              
            # Todo: Add ball tracking later      
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    ball.append(bbox)
                
            
        if stub_path is not None:
            with open(person_path,'wb') as f:
                pickle.dump(person_dict,f)
                
            with open(ball_path,'wb') as f:
                pickle.dump(ball,f)     
            
        return (person_dict, ball)
    
    
    
    def draw_annotations(self,video_frames, ball, player_dict,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            if frame_num % 25 == 0:
                print(f"Drawing frame {frame_num} of {len(video_frames)}")
            frame = frame.copy()

    

            # Draw Players
            for trackId, player in player_dict.items():
            
                if len(player.bboxes) <= frame_num:
                    continue
                if type(player) == Player:
                    frame = self.draw_ellipse(frame, player.bboxes[frame_num],(0,255,0), player.trackId)
                elif type(player) == Keeper:
                    frame = self.draw_ellipse(frame, player.bboxes[frame_num],(255,0,0), player.trackId)
                else:
                    frame = self.draw_ellipse(frame, player.bboxes[frame_num],(0,0,255), player.trackId)


            # Draw Team Ball Control

            output_video_frames.append(frame)

        return output_video_frames
    
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        if bbox is None:
            return frame
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame