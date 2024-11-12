import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image
import av



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

def reset_session_state():
    st.session_state.counter = 0
    st.session_state.stage = "down"
    st.session_state.elapsed_time = 0
    st.session_state.exercise_completed = False

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
        'message': "You seem tired. Consider taking a refreshing walk instead of intense exercise.",
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
        elif exercise_choice == 'Burpees':
            vis_point = detect_burpee(landmarks)
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
    if exercise_choice in ['Tree Pose', 'Plank']:
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

class VideoProcessor:
    def __init__(self, exercise_choice):
        self.exercise_choice = exercise_choice
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_frame = process_frame(img, self.exercise_choice, self.pose)
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

def main():
    st.title("Exercise Detection App")
    
    # Create two columns for layout
    sidebar_col, main_col = st.columns([1, 3])
    
    with st.sidebar:
        st.header("Settings")
        
        # Mood selection
        mood = st.selectbox(
            "How are you feeling today?",
            ['happy', 'sad', 'tired'],
            on_change=reset_session_state
        )
        
        # Display mood-based message
        if isinstance(MOOD_EXERCISES[mood]['message'], list):
            message = np.random.choice(MOOD_EXERCISES[mood]['message'])
        else:
            message = MOOD_EXERCISES[mood]['message']
        st.info(message)
        
        # Exercise selection
        available_exercises = MOOD_EXERCISES[mood]['exercises']
        if available_exercises:
            exercise_choice = st.selectbox(
                "Choose your exercise",
                available_exercises,
                on_change=reset_session_state
            )
        else:
            st.warning("Based on your mood, we recommend taking a break or going for a light walk.")
            return
        
        # Exercise target input
        if exercise_choice in ['Tree Pose', 'Plank', 'Side Plank']:
            target = st.slider('Target time (seconds)', 10, 300, 60)
            st.caption("Hold the position to accumulate time")
        else:
            target = st.slider('Target repetitions', 5, 50, 10)
            st.caption("Complete the full movement to count a rep")

        # Reset button
        if st.button('Reset Exercise', use_container_width=True):
            reset_session_state()
            st.rerun()

    # Main content area
    with main_col:
        # Exercise instructions tab
        instructions = {
            'Bicep Curls': {
                'steps': [
                    "Stand with feet shoulder-width apart",
                    "Hold weights at your sides, palms facing forward",
                    "Keeping upper arms still, curl weights to shoulders",
                    "Lower back down with control"
                ],
                'tips': "Keep your core tight and avoid swinging your body"
            },
            'Pushups': {
                'steps': [
                    "Start in high plank position",
                    "Lower chest to ground by bending elbows",
                    "Push back up to starting position",
                    "Maintain straight body alignment"
                ],
                'tips': "Keep your core engaged and elbows at 45° angle"
            },
            'Tree Pose': {
                'steps': [
                    "Stand on one leg",
                    "Place other foot on inner thigh or calf",
                    "Bring hands to prayer position",
                    "Fix gaze on steady point"
                ],
                'tips': "Avoid placing foot on knee joint"
            },
            'Plank': {
                'steps': [
                    "Forearms on ground, elbows under shoulders",
                    "Body in straight line from head to heels",
                    "Hold position while breathing steadily",
                    "Keep core engaged"
                ],
                'tips': "Don't let hips sag or lift"
            },
            'Jumping/Skipping': {
                'steps': [
                    "Start with feet together",
                    "Jump with both feet",
                    "Land softly on balls of feet",
                    "Maintain steady rhythm"
                ],
                'tips': "Land quietly and keep breathing steady"
            },
            'Squats': {
                'steps': [
                    "Feet shoulder-width apart",
                    "Lower hips back and down",
                    "Keep knees aligned with toes",
                    "Push through heels to stand"
                ],
                'tips': "Keep chest up and core engaged"
            }
        }

        # Display exercise instructions
        with st.expander("Exercise Instructions", expanded=True):
            if exercise_choice in instructions:
                st.subheader(f"How to do {exercise_choice}")
                for i, step in enumerate(instructions[exercise_choice]['steps'], 1):
                    st.write(f"{i}. {step}")
                st.info(f"💡 Pro tip: {instructions[exercise_choice]['tips']}")

        # WebRTC configuration
        rtc_configuration = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }

        # Create WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="exercise-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,  # Using the dictionary directly
            video_processor_factory=lambda: VideoProcessor(exercise_choice),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )       


        # Create columns for metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            if exercise_choice in ['Tree Pose', 'Plank', 'Side Plank']:
                st.metric("Time", f"{int(st.session_state.elapsed_time)} seconds")
            else:
                st.metric("Repetitions", st.session_state.counter)
        
        with metric_col2:
            if exercise_choice in ['Tree Pose', 'Plank', 'Side Plank']:
                st.metric("Target", f"{target} seconds")
            else:
                st.metric("Target", f"{target} reps")
        
        with metric_col3:
            if exercise_choice in ['Tree Pose', 'Plank', 'Side Plank']:
                progress = min(st.session_state.elapsed_time / target, 1.0)
                percentage = int(progress * 100)
                st.metric("Progress", f"{percentage}%")
            else:
                progress = min(st.session_state.counter / target, 1.0)
                percentage = int(progress * 100)
                st.metric("Progress", f"{percentage}%")

        # Progress bar
        st.progress(progress)

        # Celebration message
        if progress >= 1.0:
            st.balloons()
            st.success("🎉 Congratulations! You've reached your target!")
            if st.button("Start Another Set", use_container_width=True):
                reset_session_state()
                st.rerun()

if __name__ == '__main__':
    main()
