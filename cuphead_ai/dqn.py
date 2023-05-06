from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from collections import deque
import random
import time
import numpy as np
from ultralytics import YOLO
from PIL import ImageGrab
import cv2
import os
import pyautogui
from datetime import datetime
import pygetwindow as gw

def capture_screen(window_title):
    window = gw.getWindowsWithTitle(window_title)[0]
    screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
    screen = np.array(screenshot)
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    return screen


def is_player_dead(screen, threshold=0.8):
    retry_button = cv2.imread('retry_button.png', cv2.IMREAD_GRAYSCALE)
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    
    result = cv2.matchTemplate(screen_gray, retry_button, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    return max_val > threshold

class ObjectRecognition():
    def __init__(self, model, bounding_box, show_results, conf_threshold):
        self.model = model
        self.bounding_box = bounding_box
        self.curr_screen = None
        self.show_results = show_results
        self.conf_threshold = conf_threshold

    def get_screen_data(self):
        last_time = time.time()
        self.curr_screen = np.array(ImageGrab.grab(bbox=self.bounding_box))
        results = self.model.predict(cv2.cvtColor(self.curr_screen, cv2.COLOR_BGR2RGB), conf=self.conf_threshold,
                                show=self.show_results, device=0)
        print('Time taken for OR model to get data from screen: ', str(time.time() - last_time))
        return results[0].boxes.boxes


REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
TARGET_UPDATE_FREQ = 5

class Environment:
    def __init__(self, objrecognition, movement):
        self.player_position = None
        self.boss_position = None
        self.objrecognition = objrecognition
        self.movement = movement
        self.player_label = 1
        self.boss_label = 2
        self.ACTION_SPACE_SIZE = 3
        
    def is_overlapping(self, objects):
        found_overlaps = []
        for obj in objects:
            if obj is None:
                return False
        # Loop through each object that hasn't been compared yet and check for overlap with other objects
        compared = set()
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:
                    continue
                if (i, j) in compared:
                    continue
                # Check if the bounding boxes intersect
                x1_min, y1_min = obj1[0], obj1[1]
                x1_max, y1_max = obj1[2], obj1[3]
                x2_min, y2_min = obj2[0], obj2[1]
                x2_max, y2_max = obj2[2], obj2[3]
                if (x1_min <= x2_max and x1_max >= x2_min and
                        y1_min <= y2_max and y1_max >= y2_min):
                    found_overlaps.append((i, j))

        # Return whether any overlaps have been found.
        if len(found_overlaps) > 0:
            return True
        else:
            return False
    
    def step(self, action):
        if action == 0:  # Move Left
            self.movement.move_left()
        elif action == 1:  # Move Right
            self.movement.move_right()
        elif action == 2:  # Duck
            if (self.player_position is None) or (np.array_equal(self.player_position, np.array([-1, -1, -1, -1, -1]))):
                self.movement.clear_movement()
            else:
                self.movement.duck()
        
        reward = self.reward()
        new_state = self.get_state()
        done = self.check_terminal_state()
        
        return new_state, reward, done

    def get_state(self):
        self.capture_state()
        print("player pos:", self.player_position)
        print("boss pos:", self.boss_position)
        
        if self.player_position is None:
            self.player_position = np.array([-1, -1, -1, -1, -1])
            #self.player_position = None
            print('Could not find player.')
        if self.boss_position is None:
            #self.boss_position = None
            print('Could not find player.')
            self.boss_position = np.array([-1, -1, -1, -1, -1])
            
        return np.concatenate([self.player_position, self.boss_position])
    
    def capture_state(self):
        obj_recognition_output = self.objrecognition.get_screen_data()
        print("OR output:",obj_recognition_output)

        # Create dictionaries for keeping track of highest accuracy player / boss. initializing to -1 for sorting purposes.
        highest_accuracy = {self.player_label: -1, self.boss_label: -1}
        highest_accuracy_objdata = {self.player_label: None, self.boss_label: None}

        # Loop through objects detected by the object recognition model
        for obj in obj_recognition_output:
            x, y, w, h, accuracy, label = obj.tolist()

            # Update positions/accuracy for obj and find most accurate for ea label.
            if label == self.player_label:
                if highest_accuracy[label] < accuracy:
                    highest_accuracy[label] = accuracy
                    highest_accuracy_objdata[label] = np.array([x, y, w, h, accuracy])
            elif label == self.boss_label:
                if highest_accuracy[label] < accuracy:
                    highest_accuracy[label] = accuracy
                    highest_accuracy_objdata[label] = np.array([x, y, w, h, accuracy])
                    
         # Update the player and boss positions with the highest accuracy objects
        self.player_position = highest_accuracy_objdata[self.player_label]
        self.boss_position = highest_accuracy_objdata[self.boss_label]

    def check_terminal_state(self):
        if np.array_equal(self.player_position, np.array([-1, -1, -1, -1, -1])):
            self.movement.clear_movement()
            screen = capture_screen('Cuphead')
            # If the player is dead, break the loop
            if is_player_dead(screen):
                print("FOUND RETRY FOUND RETRY.")
                return True
            else:
                print("DID NOT FIND RETRY")
                return False
            
        elif (self.player_position is None):
            self.movement.clear_movement()
            screen = capture_screen('Cuphead')
            # If the player is dead, break the loop
            if is_player_dead(screen):
                print("FOUND RETRY FOUND RETRY.")
                return True
            else:
                print("DID NOT FIND RETRY")
                return False
        else:
            print('returning False for done, player is on screen')
            return False

    def reward(self):
        if self.is_overlapping([self.player_position, self.boss_position]):
            return -15
        else:
            return 1

    def reset(self):
        self.movement.clear_movement()
        self.player_position = None
        self.boss_position = None
        self.alive = True
        time.sleep(1.5)
        print('Resetting environment... PRESSING RETRY')
        self.press_retry()
        time.sleep(1.5)
        return self.get_state()

    def press_retry(self):
        self.movement.retry_level()
        

class DQN:
    def __init__(self):
        # The model being used for fitting
        self.model = self.create_model(10, 3)

        # The model being used for prediction
        self.target_model = self.create_model(10, 3)
        self.target_model.set_weights(self.model.get_weights())

        # Initialize the replay memory
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # ????
        self.target_update_counter = 0

    def create_model(self, input_shape, action_shape):
        model = Sequential()

        model.add(Dense(64, input_shape=[input_shape,]))
        model.add(Activation("relu"))

        model.add(Dense(32))
        model.add(Activation("relu"))

        model.add(Dropout(0.2))

        model.add(Dense(action_shape, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))

        return model

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)
        self.target_model.load_weights(weights_path)
        print(f"Weights loaded from {weights_path}")
    

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        # Predict the Q-values for the given state using the model
        return self.model.predict(state.reshape(-1, *state.shape))[0]

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        if self.target_update_counter > TARGET_UPDATE_FREQ:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        else:
            self.target_update_counter += 1


class CupHeadMovement:
    def __init__(self):
        self.is_moving_left = False
        self.is_moving_right = False
        self.is_moving_up = False
        self.is_ducking = False

    # Used to reset the movement when switching between player directions
    def clear_movement(self):
        if (self.is_moving_left):
            pyautogui.keyUp('left')
            self.is_moving_left = False
        if (self.is_moving_right):
            pyautogui.keyUp('right')
            self.is_moving_right = False
        if (self.is_moving_up):
            pyautogui.keyUp('up')
            self.is_moving_up = False
        if (self.is_ducking):
            pyautogui.keyUp('down')
            self.is_ducking = False

    def move_left(self):
        if (not self.is_moving_left):
            self.clear_movement()
            pyautogui.keyDown('left')
            self.is_moving_left = True

    def move_right(self):
        if (not self.is_moving_right):
            self.clear_movement()
            pyautogui.keyDown('right')
            self.is_moving_right = True

    def move_up(self):
        if (not self.is_moving_up):
            self.clear_movement()
            pyautogui.keyDown('up')
            self.isq_moving_up = True

    def duck(self):
        if (not self.is_ducking):
            self.clear_movement()
            pyautogui.keyDown('down')
            self.is_ducking = True

    def retry_level(self):
        pyautogui.press('enter')

WEIGHTS_SAVE_FREQ = 20
WEIGHTS_SAVE_PATH = "weights/"
EPISODES = 1000
EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 0.90
MIN_EPSILON =  0.01
print('loading weights...')
#Set current working directory and setup initialize obj rec
HOME = os.getcwd()
model = YOLO(f'{HOME}/weights/best.pt')

print('done loading weights.')

print('building obj rec')
objrecognition = ObjectRecognition(model, (0, 40, 640, 480), True, 0.2)

print('building player controller')
movement_controller = CupHeadMovement()
print('building simulated env')
env = Environment(objrecognition, movement_controller)

print('building DQN Agent')
agent = DQN()

saved_weights_path = ""

#agent.load_weights(saved_weights_path)


print('starting episodes')
for episode in range(EPISODES):
    movement_controller.clear_movement()
    current_state = env.reset()
    episode_reward = 0
    step = 1
    done = False

    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))

    while not done:

        
        if np.random.random() < epsilon:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)
        else:
            action = np.argmax(agent.get_qs(current_state, step))

        if (env.player_position is None):
            movement_controller.clear_movement().clear_movement()

        new_state, reward, done = env.step(action)

        if (env.player_position is None):
            movement_controller.clear_movement().clear_movement()
        
        episode_reward += reward

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train()

        current_state = new_state
        step += 1

    movement_controller.clear_movement()
    print('TERMINAL STATE REACHED, ENDING EPISODE!!!')
    print('TERMINAL STATE REACHED, ENDING EPISODE!!!')
    print('TERMINAL STATE REACHED, ENDING EPISODE!!!')
    print('Total reward was:', episode_reward)
    
    if episode % WEIGHTS_SAVE_FREQ == 0:
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        weights_filename = f"weights_{episode}_{current_datetime}.h5"
        agent.model.save_weights(os.path.join(WEIGHTS_SAVE_PATH, weights_filename))
        print(f"Saved weights at episode {episode} with timestamp {current_datetime}")
