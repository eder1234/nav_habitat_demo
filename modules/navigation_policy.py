
from modules.visual_memory import VisualMemory
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import os
from scipy.spatial.transform import Rotation as R
import numpy as np

class KeyBindings:
    FORWARD_KEY = "w"
    LEFT_KEY = "a"
    RIGHT_KEY = "d"
    FINISH_KEY = "f"
    MOVE_KEY = "m"
    RESET_VM_KEY = "k"
    NEXT_VM_KEY = "0"


class NavigationPolicy:
    def __init__(self, config=None, registration_result=None, visual_memory=None):
        self.config = config
        self.registration_result = registration_result
        self.fit_threshold = config['navigation']['fit_threshold']
        self.forward_threshold = config['navigation']['forward_threshold']
        self.lateral_threshold = config['navigation']['lateral_threshold']
        self.yaw_threshold = config['navigation']['yaw_threshold']
        self.vm_path = config['paths']['VM_PATH']
        self.visual_memory = visual_memory or VisualMemory(config)

    def print_success(self, message):
        green = "\033[92m"
        reset = "\033[0m"
        print(f"{green}Success: {message}{reset}")

    def print_warning(self, message):
        red = "\033[91m"
        reset = "\033[0m"
        print(f"{red}Warning: {message}{reset}")

    def determine_bot_action(self, registration_result):
        """
        Determine the action a bot should take based on the transformation matrix.
        Args:
        T (np.array): A 4x4 transformation matrix containing rotation and translation.
        Returns:
        str: The action the bot should take: 'Move Forward', 'Turn Right', 'Turn Left', or 'Stop'.
        """
        self.registration_result = registration_result

        if self.registration_result.fitness < self.fit_threshold:
            self.print_warning("Warning: Regitration failed. Fitness score: {}".format(self.registration_result.fitness))
        else:
            self.print_success("Registration successful. Fitness score: {}".format(self.registration_result.fitness))

        # Extract the translation vector and Euler angles
        print('Processing action')
        T = np.copy(self.registration_result.transformation)
        translation = T[0:3, 3]
        rotation_matrix = T[0:3, 0:3]
        euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
        
        print(f'Translation: {translation}')
        print(f'Angles (xyz): {euler_angles}')

        # Check translation for forward/backward movement
        if translation[0] < -self.forward_threshold:
            action_forward = 'Move Forward'
        else:
            action_forward = 'Stop'  # If the bot is close enough to the target

        # Check lateral translation and yaw angle for turning
        if translation[1] < -self.lateral_threshold or euler_angles[2] < -self.yaw_threshold:
            action_turn = 'Turn Right'
        elif translation[1] > self.lateral_threshold or euler_angles[2] > self.yaw_threshold:
            action_turn = 'Turn Left'
        else:
            action_turn = None  # No turn is needed if within thresholds

        # Combine actions: prioritize turning over moving forward
        if action_turn:
            return action_turn
        else:
            return action_forward
        
    def handle_keystroke(self, keystroke, vm_image_index, registration_result):
        if keystroke == ord(KeyBindings.MOVE_KEY):
            computed_action = 'Stop'

            computed_action = self.determine_bot_action(registration_result)

            if computed_action == 'Move Forward':
                action = HabitatSimActions.move_forward
            elif computed_action == 'Turn Right':
                action = HabitatSimActions.turn_right
            elif computed_action == 'Turn Left':
                action = HabitatSimActions.turn_left
            elif computed_action == 'Stop':
                vm_image_index = (vm_image_index + 1) % len(os.listdir(self.vm_path + "color/"))
                self.visual_memory.display_visual_memory( vm_image_index)
                return vm_image_index, None  # No action to execute

        elif keystroke == ord(KeyBindings.FORWARD_KEY):
            action = HabitatSimActions.move_forward

        elif keystroke == ord(KeyBindings.LEFT_KEY):
            action = HabitatSimActions.turn_left

        elif keystroke == ord(KeyBindings.RIGHT_KEY):
            action = HabitatSimActions.turn_right

        elif keystroke == ord(KeyBindings.FINISH_KEY):
            print("Finishing the episode.")
            return vm_image_index, "finish"  # Signal to finish the episode

        elif keystroke == ord(KeyBindings.RESET_VM_KEY):
            vm_image_index = 0
            self.visual_memory.display_visual_memory(vm_image_index)
            return vm_image_index, None  # No action to execute

        elif keystroke == ord(KeyBindings.NEXT_VM_KEY):
            vm_image_index = (vm_image_index + 1) % len(os.listdir(self.vm_path + "color/"))
            self.visual_memory.display_visual_memory(vm_image_index)
            return vm_image_index, None  # No action to execute

        else:
            return vm_image_index, None  # No action for unrecognized keystrokes

        # For actions that involve moving the agent
        return vm_image_index, action