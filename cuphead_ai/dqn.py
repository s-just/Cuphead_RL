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

### Notes:
### If I ever come back to this project I would like to reimplement obj recognition with higher accuracies for cuphead being hit and overlapped with the slime boss.
### Also I would use my GPU for the DQN model so training could be improved.
### Also the reward function could be improved to win with less powerful weapons.


# Capture the cuphead window for pressing retry without exiting to map.
def capture_screen(window_title):
    window = gw.getWindowsWithTitle(window_title)[0]
    screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
    screen = np.array(screenshot)
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    return screen

# Determines if the retry button is on the screen and is also highlighted.
def is_player_dead(screen, threshold=0.8):
    retry_button = cv2.imread('retry_button.png')  # Removed cv2.IMREAD_GRAYSCALE
    # Removed screen_gray conversion since we need the color for highlight detection
    result = cv2.matchTemplate(screen, retry_button, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val > threshold

# The main Object Recognition class
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
        #results = self.model.predict(cv2.cvtColor(self.curr_screen, cv2.COLOR_BGR2RGB), conf=self.conf_threshold,
        #                        show=self.show_results, device=0)
        results = self.model.predict(cv2.cvtColor(self.curr_screen, cv2.COLOR_BGR2RGB), conf=self.conf_threshold,device=0)
        #print('Time taken for OR model to get data from screen: ', str(time.time() - last_time))
        #print(results[0].boxes.boxes)
        return results[0].boxes.boxes

# Basic DQN settings
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 500
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
TARGET_UPDATE_FREQ = 5

# The environment that is being built from object recognition
class Environment:
    def __init__(self, objrecognition, movement):
        self.player_position = None
        self.player_facing = "right"
        self.boss_position = None
        self.objrecognition = objrecognition
        self.movement = movement
        self.player_label = 1
        self.boss_label = 2
        self.missile_label = 0
        self.ACTION_SPACE_SIZE = 3
        self.current_missiles = [] # Originally I was tracking missiles but ended up changing my reward function
        
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

    # Check if the player is facing the boss (important for reward system)
    def is_player_facing_boss(self):
        if self.player_facing == "right" and self.player_position[0] < self.boss_position[0]:
            return True
        elif self.player_facing == "left" and self.player_position[0] > self.boss_position[0]:
            return True
        else:
            return False

    def step(self, action):
        if action == 0:  # Move Left
            if (self.is_player_facing_boss()):
                self.movement.clear_movement()
                pyautogui.press('v')
            self.movement.move_left()
            self.player_facing = "left"
        elif action == 1:  # Move Right
            if (self.is_player_facing_boss()):
                self.movement.clear_movement()
                pyautogui.press('v')
            self.movement.move_right()
            self.player_facing = "right"
        elif action == 2:  # Duck
            if (self.player_position is None) or (np.array_equal(self.player_position, np.array([-1, -1, -1, -1, -1]))):
                self.movement.clear_movement()
            else:
                self.movement.duck()
        
        reward = self.reward(action)
        new_state = self.get_state()
        done = self.check_terminal_state()
        
        return new_state, reward, done

    # Grabs the state using the capture_state method and also handles situations where the player is not detected on screen (this is important as the shape to the DQN must be consistent).
    def get_state(self):
        self.capture_state()
        #print("player pos:", self.player_position)
        #print("boss pos:", self.boss_position)
        #print("missiles list:", self.current_missiles)
        
        if self.player_position is None:
            self.player_position = np.array([-1, -1, -1, -1, -1])
            #self.player_position = None
            #print('Could not find player.')
        if self.boss_position is None:
            #self.boss_position = None
            #print('Could not find player.')
            self.boss_position = np.array([-1, -1, -1, -1, -1])
            
        return np.concatenate([self.player_position, self.boss_position])

    # Capture the state using the object recognition class and only take the highest accuracy detections.
    # This method also includes an implementation for reducing the size of the bounding boxes detected so that they can better represent the player and boss hitbox.
    def capture_state(self):
        obj_recognition_output = self.objrecognition.get_screen_data()
        #print("OR output:",obj_recognition_output)

        # Create dictionaries for keeping track of highest accuracy player / boss. initializing to -1 for sorting purposes.
        highest_accuracy = {self.player_label: -1, self.boss_label: -1}
        highest_accuracy_objdata = {self.player_label: None, self.boss_label: None}

        missiles = []
        
        w_padding_percent = 0.03  # Padding percentage, adjust this value to change the padding size
        h_padding_percent = 0.1
        # Loop through objects detected by the object recognition model
        for obj in obj_recognition_output:
            x, y, w, h, accuracy, label = obj.tolist()

            # Apply padding to the bounding box
            w_pad = int(w * w_padding_percent)
            h_pad = int(h * h_padding_percent)
            x += w_pad // 2
            y += h_pad // 2
            w -= w_pad
            h -= h_pad

            # Update positions/accuracy for obj and find most accurate for ea label.
            if label == self.player_label:
                if highest_accuracy[label] < accuracy:
                    highest_accuracy[label] = accuracy
                    highest_accuracy_objdata[label] = np.array([x, y, w, h, accuracy])
            elif label == self.boss_label:
                if highest_accuracy[label] < accuracy:
                    highest_accuracy[label] = accuracy
                    highest_accuracy_objdata[label] = np.array([x, y, w, h, accuracy])
            elif label == self.missile_label:
                if (accuracy > 0.5):
                    missiles.append(np.array([x, y, w, h, accuracy]))
        self.current_missiles = missiles
                
        # Update the player and boss positions with the highest accuracy objects
        self.player_position = highest_accuracy_objdata[self.player_label]
        self.boss_position = highest_accuracy_objdata[self.boss_label]


    # A method for checking if a single missile between the player and the boss,
    def is_missile_between(self, player, boss, missiles):
        for missile in missiles:
            if missile is None:
                return False

            # Check if the missile's x coordinates are between the player's and boss's x coordinates
            if min(player[0], boss[0]) < missile[0] < max(player[2], boss[2]) and \
               min(player[1], boss[1]) < missile[1] < max(player[3], boss[3]):
                return True

        return False

    # Check to see if thie boss is going to take damage from a missile that was previously fired.  (I stopped using this reward system)
    def check_for_boss_dmg(self):
        for missile in self.current_missiles:
            if self.is_missile_between(self.player_position, self.boss_position, self.current_missiles):
                #print('SHOOTING BOSS')
                #print('SHOOTING BOSS')
                #print('SHOOTING BOSS')
                return True
        #print('not shooting boss')
        return False

    # Check to see if the retry option is visible, which determines if the episode should be terminated
    def check_terminal_state(self):
        if np.array_equal(self.player_position, np.array([-1, -1, -1, -1, -1])):
            self.movement.clear_movement()
            screen = capture_screen('Cuphead')
            # If the player is dead, break the loop
            if is_player_dead(screen):
                print("FOUND RETRY FOUND RETRY.")
                return True
            else:
                #print("DID NOT FIND RETRY")
                return False
            
        elif (self.player_position is None):
            self.movement.clear_movement()
            screen = capture_screen('Cuphead')
            # If the player is dead, break the loop
            if is_player_dead(screen):
                print("FOUND RETRY FOUND RETRY.")
                return True
            else:
                #print("DID NOT FIND RETRY")
                return False
        else:
            #print('returning False for done, player is on screen')
            return False

    # Checks if the player is going to be colliding with the boss. Helps with reward systems.
    def is_agent_moving_towards_boss(self, action):
        if action == 1:
            if self.player_facing == "right":
                if self.is_player_facing_boss():
                    return True
        if action == 0:
            if self.player_facing == "left":
                if self.is_player_facing_boss():
                    return True
        return False
                
    # The method for determing the agent's reward. Currently ensures Cuphead maintains an optimal distance from the boss and is not being hit by the boss.
    def reward(self, action):
        damage_boss_bonus = 0

        if (self.is_player_facing_boss()):
            print('facing boss')
            damage_boss_bonus += 1.5
        else:
            print('not facing boss')
            damage_boss_bonus -= 1

        # Calculate distance between player and boss
        player_center_x = self.player_position[0] + self.player_position[2] / 2
        player_center_y = self.player_position[1] + self.player_position[3] / 2
        boss_center_x = self.boss_position[0] + self.boss_position[2] / 2
        boss_center_y = self.boss_position[1] + self.boss_position[3] / 2
        distance = np.sqrt((player_center_x - boss_center_x) ** 2 + (player_center_y - boss_center_y) ** 2)
        
        # Define distance penalty based on the calculated distance
        distance_penalty = 0
        min_distance = 150  # Minimum desired distance between the player and boss
        optimal_distance = 180  # Optimal desired distance between the player and boss

        if distance < min_distance:
            distance_penalty = -2 * (min_distance - distance) / min_distance

        # Penalty for moving towards boss within optimal distance
        moving_towards_boss_penalty = 0
        if distance < optimal_distance and self.is_agent_moving_towards_boss(action):
            print('moving towards boss within optimal distance')
            moving_towards_boss_penalty = -10

        print("moving towards boss penalty", moving_towards_boss_penalty)
        print("distance from boss:", distance," / distance penalty:", distance_penalty)
        
        if self.is_overlapping([self.player_position, self.boss_position]):
            print('TAKING DAMAGE FROM BOSS')
            return -32
        else:
            print('not taking any dmg from boss')
            return 3 + damage_boss_bonus + distance_penalty + moving_towards_boss_penalty

    # Resets the environment by pressing the retry button and waiting for the battle to start. Also resets env variables and clears movement.
    def reset(self, ignore_check):
        self.movement.clear_movement()
        self.player_position = None
        self.player_facing = "right"
        self.boss_position = None
        self.alive = True
        time.sleep(1.5)
        print('Resetting environment... PRESSING RETRY')
        self.press_retry(ignore_check)
        time.sleep(1.5)
        return self.get_state()
    
    # Handles the act of pressing retry, which is necessary since sometimes agent movement can change the menu option unintentionally and the highlighted option needs to be adjusted.
    def press_retry(self, ignore_check):
        self.movement.clear_movement()
        self.movement.clear_movement()
        self.movement.clear_movement()
        if ignore_check:
            print('ignoring retry check')
            self.movement.retry_level()
        else:
            while True:
                screen = capture_screen('Cuphead')
                if is_player_dead(screen, threshold=0.95):
                    print('found highlighted retry')
                    self.movement.retry_level()
                    break
                else:
                    print('menu bugged, finding retry')
                    pyautogui.press('up')
                    time.sleep(1)
                    
        
# The DQN used for the agent.
class DQN:
    def __init__(self):
        # The model being used for fitting
        self.model = self.create_model(10, 3)

        # The model being used for prediction
        self.target_model = self.create_model(10, 3)
        self.target_model.set_weights(self.model.get_weights())

        # Initialize the replay memory
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

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

# A class built using pyautogui, determines the movement statement and switches between them, holding down the correct key based on the state.
class CupHeadMovement:
    def __init__(self):
        self.is_moving_left = False
        self.is_moving_right = False
        self.is_moving_up = False #Not utilized
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

# Initialize some hyperparameters and save variables
WEIGHTS_SAVE_FREQ = 20
WEIGHTS_SAVE_PATH = "weights/"
EPISODES = 1000
EPSILON_START = 0.02
EPSILON_END = 0.01
EPSILON_DECAY = 0.99
MIN_EPSILON =  0.01


#Set current working directory and setup initialize obj rec
HOME = os.getcwd()
model = YOLO(f'{HOME}/weights/best.pt')

print('done loading objr weights.')

print('building obj rec')
objrecognition = ObjectRecognition(model, (0, 40, 640, 480), True, 0.2)

print('building player controller')
movement_controller = CupHeadMovement()

print('building simulated env')
env = Environment(objrecognition, movement_controller)

print('building DQN Agent')
agent = DQN()

#saved_weights_path = ""

#agent.load_weights(saved_weights_path)

# The training loop, uses epsilon to determine whether an action should be random or based on q values. The player is always firing.
print('starting episodes')
for episode in range(EPISODES):
    
        
    movement_controller.clear_movement()
    if episode == 0:
        current_state = env.reset(True)
    else:
        current_state = env.reset(False)
    episode_reward = 0
    step = 1
    done = False

    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))

    while not done:
        pyautogui.keyDown('x')
        if np.random.random() < epsilon:
            print('random action')
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)
        else:
            print('trained action')
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

    pyautogui.keyUp('x')
    
    movement_controller.clear_movement()
    print('TERMINAL STATE REACHED, ENDING EPISODE!!!')
    print('TERMINAL STATE REACHED, ENDING EPISODE!!!')
    print('TERMINAL STATE REACHED, ENDING EPISODE!!!')
    print('Total reward was:', episode_reward)

    # Save weights based on the freq variable.
    if episode % WEIGHTS_SAVE_FREQ == 0:
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        weights_filename = f"weights_{episode}_{current_datetime}.h5"
        agent.model.save_weights(os.path.join(WEIGHTS_SAVE_PATH, weights_filename))
        print(f"Saved weights at episode {episode} with timestamp {current_datetime}")
