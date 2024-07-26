from ultralytics import YOLO

print("Start")
model = YOLO('models/best.pt')  # Load model

model.predict("Input/08fd33_4.mp4", save = True)  # Inference on video (save=True to save results to 'runs/detect' folder)

print(results[0])
for box in results[0]:
    print(box)