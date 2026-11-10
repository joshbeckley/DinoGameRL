import pygame
import math

class NeuralNetworkVisualizer:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        
        # Fonts
        self.font = pygame.font.SysFont("Arial", 14, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 12)
        self.title_font = pygame.font.SysFont("Arial", 16, bold=True)
        
        # Color Palette
        self.NODE_INPUT = (96, 165, 250)     # Blue
        self.NODE_HIDDEN = (74, 222, 128)    # Green
        self.NODE_STREAM = (251, 191, 36)    # Amber
        self.NODE_OUTPUT = (248, 113, 113)   # Red
        self.ACTIVE_GLOW = (255, 255, 255)   # White highlight for firing
        self.TEXT_COLOR = (0, 0, 0)
        
        # Animation timer for firing pulse effect
        self.pulse_timer = 0

    def _get_layer_coords(self, base_x, count, start_y, total_height):
        coords = []
        spacing = min(35, total_height / max(1, count))
        for i in range(count):
            coords.append((base_x, start_y + i * spacing + spacing / 2))
        return coords

    def draw(self, screen, state_vals, active_action=None):
        """
        Draws the neural network with live state values, firing animation pulses,
        and highlights the selected action (0 = Do Nothing, 1 = Jump).
        """
        self.pulse_timer += 0.15
        pulse = (math.sin(self.pulse_timer) + 1) / 2  # Value between 0 and 1
        
        col_x = [self.x + 150, self.x + 420, self.x + 680, self.x + 920]
        
        state_labels = [
            f"1. grounded: {state_vals[0]}", 
            f"2. speed: {state_vals[1]:.1f}", 
            f"3. dist: {state_vals[2]}", 
            f"4. y_above ground: {state_vals[3]}", 
            f"5. vertical speed: {state_vals[4]:.2f}"
        ]

        start_y = self.y + 60
        layer_height = 200
        
        in_coords = self._get_layer_coords(col_x[0], 5, start_y, layer_height)
        hid_coords = self._get_layer_coords(col_x[1], 8, start_y, layer_height)
        val_coords = self._get_layer_coords(col_x[2] - 30, 4, start_y, layer_height)
        adv_coords = self._get_layer_coords(col_x[2] + 30, 4, start_y, layer_height)
        out_coords = self._get_layer_coords(col_x[3], 2, start_y, layer_height)

        # --- DRAW SYNAPSE CONNECTIONS (Highlight active propagation paths) ---
        base_line_color = (40, 50, 70)
        active_line_color = (90, 140, 180)

        # Input -> Hidden lines
        for i, (ix, iy) in enumerate(in_coords):
            # If state input is non-zero, make its connections brighter ("firing")
            is_active_input = (state_vals[i] > 0)
            for hx, hy in hid_coords:
                color = active_line_color if is_active_input else base_line_color
                pygame.draw.line(screen, color, (ix, iy), (hx, hy), 1)

        # Hidden -> Streams
        for hx, hy in hid_coords:
            for vx, vy in val_coords:
                pygame.draw.line(screen, (35, 45, 65), (hx, hy), (vx, vy), 1)
            for ax, ay in adv_coords:
                pygame.draw.line(screen, (35, 45, 65), (hx, hy), (ax, ay), 1)

        # Streams -> Output Q-Values
        for vx, vy in val_coords:
            for ox, oy in out_coords:
                pygame.draw.line(screen, (50, 35, 45), (vx, vy), (ox, oy), 1)
        for ax, ay in adv_coords:
            for i, (ox, oy) in enumerate(out_coords):
                # Highlight path to chosen action
                is_chosen = (active_action == i)
                color = (220, 100, 100) if is_chosen else (50, 35, 45)
                pygame.draw.line(screen, color, (ax, ay), (ox, oy), 2 if is_chosen else 1)

        # --- DRAW NODES & FIRING PULSES ---
        
        # 1. Inputs
        for i, (x, y) in enumerate(in_coords):
            is_active = (state_vals[i] > 0)
            node_color = (120, 190, 255) if is_active else self.NODE_INPUT
            pygame.draw.circle(screen, node_color, (int(x), int(y)), 9)
            if is_active:
                pygame.draw.circle(screen, self.ACTIVE_GLOW, (int(x), int(y)), 12, 1)
            
            txt = self.small_font.render(state_labels[i], True, self.TEXT_COLOR)
            screen.blit(txt, (x - 130, y - 6))

        # 2. Hidden (Pulse a few nodes to simulate neural firing activity)
        for idx, (x, y) in enumerate(hid_coords):
            is_pulsing = (idx % 2 == int(self.pulse_timer * 2) % 2)
            color = (110, 245, 160) if is_pulsing else self.NODE_HIDDEN
            pygame.draw.circle(screen, color, (int(x), int(y)), 7)

        # 3. Streams
        for x, y in val_coords:
            pygame.draw.circle(screen, self.NODE_STREAM, (int(x), int(y)), 6)
        for x, y in adv_coords:
            pygame.draw.circle(screen, self.NODE_STREAM, (int(x), int(y)), 6)

        # 4. Outputs (Highlight the chosen action with a glowing outer ring)
        action_labels = ["Q(s, Do Nothing)", "Q(s, Jump)"]
        for i, (x, y) in enumerate(out_coords):
            is_selected = (active_action == i)
            
            # Draw outer selection ring if it's the chosen action
            if is_selected:
                ring_radius = int(12 + pulse * 4)
                pygame.draw.circle(screen, (255, 215, 0), (int(x), int(y)), ring_radius, 2)

            pygame.draw.circle(screen, self.NODE_OUTPUT, (int(x), int(y)), 10)
            
            label_text = f"{action_labels[i]} [CHOSEN]" if is_selected else action_labels[i]
            txt_color = (255, 230, 100) if is_selected else self.TEXT_COLOR
            txt = self.font.render(label_text, True, txt_color)
            screen.blit(txt, (x + 18, y - 8))

        # --- HEADERS ---
        headers = [
            ("State Inputs (Dim: 5)", col_x[0] - 70),
            ("Hidden fc1 (256)", col_x[1] - 60),
            ("Dueling Streams (V / A)", col_x[2] - 80),
            ("Q-Values", col_x[3] - 30)
        ]
        for title, x in headers:
            t_surf = self.title_font.render(title, True, (255, 255, 255))
            screen.blit(t_surf, (x, self.y + 20))