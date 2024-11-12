import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image
import tempfile

st.set_page_config(page_title="Exercise Page")

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize session state variables
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'stage' not in st.session_state:
    st.session_state.stage = "down"
if 'elapsed_time' not in st.session_state:
    st.session_state.elapsed_time = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'exercise_completed' not in st.session_state:
    st.session_state.exercise_completed = False
if 'running' not in st.session_state:
    st.session_state.running = False

def reset_session_state():
    st.session_state.counter = 0
    st.session_state.stage = "down"
    st.session_state.elapsed_time = 0
    st.session_state.exercise_completed = False
    st.session_state.running = False


# Mood-based exercise recommendations
MOOD_EXERCISES = {
    'happy': {
        'message': "Great mood! Let's make it even better with these exercises:",
        'exercises': ['Jumping/Skipping', 'Burpees', 'Pushups', 'Squats','Bicep Curls']
    },
    'sad': {
        'message': [
            "Remember, exercise releases endorphins - nature's mood lifter!",
            "Every rep brings you closer to feeling better!",
            "You're stronger than you think - let's prove it!",
            "Taking care of your body helps take care of your mind."
        ],
        'exercises': ['Tree Pose', 'Plank', 'Squats','Lunges']
    },
    'tired': {
        'message': "You seem tired. Consider taking a refreshing walk instead of intense exercise. Walking can help boost your energy levels naturally.",
        'exercises': []
    }
}

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def detect_bicep_curl(landmarks):
    """Detect bicep curls and count reps"""
    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    
    angle = calculate_angle(shoulder, elbow, wrist)
    
    if angle > 160:
        st.session_state.stage = "down"
    elif angle < 30 and st.session_state.stage == 'down':
        st.session_state.stage = "up"
        st.session_state.counter += 1
        
    return angle, elbow

def detect_pushup(landmarks):
    """Detect pushups and count reps"""
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    
    angle = calculate_angle(shoulder, elbow, wrist)
    
    if angle > 160:
        st.session_state.stage = "up"
    elif angle < 90 and st.session_state.stage == 'up':
        st.session_state.stage = "down"
        st.session_state.counter += 1
        
    return angle, elbow

def detect_tree_pose(landmarks):
    """Detect Tree Pose (Vrksasana)"""
    current_time = time.time()

    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    
    angle = calculate_angle(hip, knee, ankle)
    
    if angle < 80:
        if st.session_state.stage != "detected":
            st.session_state.stage = "detected"
            st.session_state.start_time = current_time
        st.session_state.elapsed_time = min(st.session_state.elapsed_time + 
                                          (current_time - st.session_state.start_time), 720)
    else:
        st.session_state.stage = "not detected"
        st.session_state.start_time = current_time

    return angle

def detect_jumping(landmarks):
    """Detect jumping or rope skipping and count reps"""
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    hip_height = (left_hip + right_hip) / 2
    ground_level = 0.7
    
    if hip_height < ground_level:
        if st.session_state.stage == "down":
            st.session_state.counter += 1
            st.session_state.stage = "up"
    else:
        st.session_state.stage = "down"
    
    return hip_height, [0.5, hip_height]

def detect_squat(landmarks):
    """Detect squats and count reps"""
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    
    angle = calculate_angle(hip, knee, ankle)
    
    if angle > 160:
        st.session_state.stage = "up"
    elif angle < 90 and st.session_state.stage == 'up':
        st.session_state.stage = "down"
        st.session_state.counter += 1
        
    return angle, knee

def detect_plank(landmarks):
    """Detect plank hold"""
    current_time = time.time()

    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    
    shoulder_elbow_dist = np.linalg.norm(np.array(shoulder) - np.array(elbow))
    shoulder_hip_dist = np.linalg.norm(np.array(shoulder) - np.array(hip))
    
    if abs(shoulder_elbow_dist - shoulder_hip_dist) < 0.1:
        if st.session_state.stage != "detected":
            st.session_state.stage = "detected"
            st.session_state.start_time = current_time
        st.session_state.elapsed_time = min(st.session_state.elapsed_time + 
                                          (current_time - st.session_state.start_time), 360)
    else:
        st.session_state.stage = "not detected"
        st.session_state.start_time = current_time

def detect_lunge(landmarks):
    """Detect lunges and count reps"""
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    
    angle = calculate_angle(hip, knee, ankle)
    
    if angle > 160:
        st.session_state.stage = "up"
    elif angle < 90 and st.session_state.stage == 'up':
        st.session_state.stage = "down"
        st.session_state.counter += 1
        
    return angle, knee

def detect_burpee(landmarks):
    """Detect burpees and count reps"""
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    
    if shoulder > hip:
        if st.session_state.stage == "down":
            st.session_state.stage = "up"
            st.session_state.counter += 1
    else:
        st.session_state.stage = "down"
        
    return [shoulder, hip]

def detect_sit_up(landmarks):
    """Detect sit-ups and count reps"""
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    
    if shoulder < hip:
        if st.session_state.stage == "down":
            st.session_state.stage = "up"
            st.session_state.counter += 1
    else:
        st.session_state.stage = "down"
        
    return [shoulder, hip]

def detect_side_plank(landmarks):
    """Detect side plank hold"""
    current_time = time.time()

    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    
    shoulder_hip_dist = np.linalg.norm(np.array(shoulder) - np.array(hip))
    hip_ankle_dist = np.linalg.norm(np.array(hip) - np.array(ankle))
    
    if abs(shoulder_hip_dist - hip_ankle_dist) < 0.1:
        if st.session_state.stage != "detected":
            st.session_state.stage = "detected"
            st.session_state.start_time = current_time
        st.session_state.elapsed_time = min(st.session_state.elapsed_time + 
                                          (current_time - st.session_state.start_time), 240)
    else:
        st.session_state.stage = "not detected"
        st.session_state.start_time = current_time

def process_frame(frame, exercise_choice, pose):
    """Process each frame and return the annotated image"""
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Make detection
    results = pose.process(image)
    
    # Convert back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        landmarks = results.pose_landmarks.landmark
        
        # Detect exercise based on choice
        if exercise_choice == 'Tree Pose':
            angle = detect_tree_pose(landmarks)
        elif exercise_choice == 'Plank':
            detect_plank(landmarks)
        elif exercise_choice == 'Side Plank':
            detect_side_plank(landmarks)
        elif exercise_choice == 'Burpees':
            vis_point = detect_burpee(landmarks)
        elif exercise_choice == 'Sit-ups':
            vis_point = detect_sit_up(landmarks)
        else:
            if exercise_choice == 'Bicep Curls':
                angle, vis_point = detect_bicep_curl(landmarks)
            elif exercise_choice == 'Pushups':
                angle, vis_point = detect_pushup(landmarks)
            elif exercise_choice == 'Jumping/Skipping':
                _, vis_point = detect_jumping(landmarks)
            elif exercise_choice == 'Squats':
                angle, vis_point = detect_squat(landmarks)
            elif exercise_choice == 'Lunges':
                angle, vis_point = detect_lunge(landmarks)
                
            # Visualize angle
            cv2.putText(image, f'{angle:.1f}', 
                        tuple(np.multiply(vis_point, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
    except:
        pass
    
    # Draw status box
    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
    
    # Display exercise name
    cv2.putText(image, exercise_choice, (10,25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
    
    # Display counter or timer
    if exercise_choice in ['Tree Pose', 'Plank', 'Side Plank']:
        time_text = f'Time: {int(st.session_state.elapsed_time)} seconds'
        cv2.putText(image, time_text, (10,50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
    else:
        counter_text = f'Count: {st.session_state.counter}'
        cv2.putText(image, counter_text, (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
    
    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )
    
    return image

def get_available_cameras():
    """Detect available cameras in the system"""
    available_cameras = []
    for i in range(3):  # Check first 3 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Read a frame to get resolution
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                available_cameras.append({
                    'index': i,
                    'name': f'Camera {i}',
                    'resolution': f'{width}x{height}'
                })
            cap.release()
    return available_cameras

def main():
    st.title("Click Start if You are not able to see yourself press Reset")
    
    # Sidebar for user input
    st.sidebar.header("Settings")
    
    # Mood selection
    mood = st.sidebar.selectbox(
        "How are you feeling today?",
        ['happy', 'sad', 'tired'],
        on_change=reset_session_state
    )
    
    # Display mood-based message
    if isinstance(MOOD_EXERCISES[mood]['message'], list):
        message = np.random.choice(MOOD_EXERCISES[mood]['message'])
    else:
        message = MOOD_EXERCISES[mood]['message']
    st.sidebar.info(message)
    
    # Exercise selection
    available_exercises = MOOD_EXERCISES[mood]['exercises']
    if available_exercises:
        exercise_choice = st.sidebar.selectbox(
            "Choose your exercise",
            available_exercises,
            on_change=reset_session_state
        )
    else:
        st.warning("Based on your mood, we recommend taking a break or going for a light walk.")
        return
    
    # Exercise target input
    if exercise_choice in ['Tree Pose', 'Plank', 'Side Plank']:
        target = st.sidebar.slider('Target time (seconds)', 10, 300, 60)
    else:
        target = st.sidebar.slider('Target repetitions', 5, 50, 10)
    
    # Camera input selection
    camera_input = st.sidebar.radio(
        "Select input source",
        ('Camera', 'Upload Video')
    )
    
    # Main content
    if camera_input == 'Camera':
        # Get available cameras
        available_cameras = get_available_cameras()
        
        if not available_cameras:
            st.error("No cameras detected on your system.")
            return
        
        # Create camera selection options
        camera_options = {
            f"{cam['name']} ({cam['resolution']})": cam['index'] 
            for cam in available_cameras
        }
        
        # Camera selection dropdown
        selected_camera = st.sidebar.selectbox(
            "Choose camera",
            options=list(camera_options.keys())
        )
        
        # Get selected camera index
        camera_index = camera_options[selected_camera]
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        
        # Add camera settings
        if cap.isOpened():
            # Get current camera settings
            current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Add resolution selection
            resolution_options = [
                f"{current_width}x{current_height} (Current)",
                "640x480",
                "1280x720",
                "1920x1080"
            ]
            
            selected_resolution = st.sidebar.selectbox(
                "Select resolution",
                resolution_options
            )
            
            # Apply selected resolution
            if selected_resolution != resolution_options[0]:
                width, height = map(int, selected_resolution.split('x'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Add FPS selection
            current_fps = int(cap.get(cv2.CAP_PROP_FPS))
            fps_options = [15, 30, 60]
            selected_fps = st.sidebar.selectbox(
                "Select FPS",
                fps_options,
                index=fps_options.index(30) if 30 in fps_options else 0
            )
            cap.set(cv2.CAP_PROP_FPS, selected_fps)
            
    else:
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            st.warning("Please upload a video file.")
            return
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        # Create placeholder for video feed
        stframe = st.empty()
        
        # Start/Stop button
        if not st.session_state.running:
            if st.button('Start Exercise'):
                st.session_state.running = True
        else:
            if st.button('Stop Exercise'):
                st.session_state.running = False
        
        while cap.isOpened() and st.session_state.running:
            ret, frame = cap.read()
            
            if not ret:
                st.error("Failed to read from video source")
                break
            
            # Process frame
            frame = process_frame(frame, exercise_choice, pose)
            
            # Display frame
            stframe.image(frame, channels="BGR")
            
            # Check if target is reached
            if exercise_choice in ['Tree Pose', 'Plank', 'Side Plank']:
                if st.session_state.elapsed_time >= target:
                    st.success(f"Great job! You've held the {exercise_choice} for {target} seconds!")
                    st.session_state.running = False
                    break
            else:
                if st.session_state.counter >= target:
                    st.success(f"Congratulations! You've completed {target} {exercise_choice}!")
                    st.session_state.running = False
                    break
        
        cap.release()

    # Reset button
    if st.button('Reset'):
        st.session_state.counter = 0
        st.session_state.stage = "down"
        st.session_state.elapsed_time = 0
        st.session_state.exercise_completed = False
        st.session_state.running = False
        st.experimental_set_query_params(reload="true")
st.set_option('client.showErrorDetails', False)
if __name__ == '__main__':
    main()
