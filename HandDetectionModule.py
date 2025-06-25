from typing import Optional
import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """A class for detecting and tracking hands in images using MediaPipe."""

    DEFAULT_MAX_HANDS = 2
    DEFAULT_DETECTION_CONFIDENCE = 0.5
    DEFAULT_TRACKING_CONFIDENCE = 0.5

    def __init__(
            self,
            static_image_mode: bool = False,
            max_hands: int = DEFAULT_MAX_HANDS,
            detection_confidence: float = DEFAULT_DETECTION_CONFIDENCE,
            tracking_confidence: float = DEFAULT_TRACKING_CONFIDENCE
    ):
        """Initialize the HandDetector.
        
        Args:
            static_image_mode: If True, treats input images as independent frames
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum confidence for hand detection
            tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.results = None

    def find_hands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """Detect hands in the input frame and optionally draw landmarks.
        
        Args:
            frame: Input image frame in BGR format
            draw: If True, draws hand landmarks on the frame
            
        Returns:
            Frame with optionally drawn hand landmarks
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self._draw_hand_landmarks(frame, hand_landmarks)

        return frame

    def _draw_hand_landmarks(self, frame: np.ndarray, hand_landmarks: mp.solutions.hands.HandLandmark) -> None:
        """Draw hand landmarks and connections on the frame.
        
        Args:
            frame: Image frame to draw on
            hand_landmarks: Detected hand landmarks to draw
        """
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

    def find_position(self, img, draw=True) -> list:
        """Find the positions of hand landmarks in the image.
        
        Args:
            img: Input image frame
            draw: If True, draws circles at landmark positions
            
        Returns:
            List of landmark positions [id, x, y] or empty list if no hands detected
        """
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[0]
            for id_, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id_, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return lm_list
