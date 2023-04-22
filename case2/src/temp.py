import pygame
import random

pygame.init()

# Set up the display
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Raffle")

# Load the background images
bg_image = pygame.image.load("background.png")
reroll_image = pygame.image.load("reroll.png")

# Load the font
FONT_SIZE = 30
font = pygame.font.SysFont(None, FONT_SIZE)

# Define the function for selecting a winner
def select_winner(names, locations):
    if len(names) == 0:
        return None
    else:
        winner = random.choice(names)
        names.remove(winner)
        location = random.choice(locations)
        locations.remove(location)
        return winner, location

# Set up the initial state of the raffle
names = ["Alice", "Bob", "Charlie", "Dave", "Eve", "Frank"]
locations = [(100, 100), (200, 100), (300, 100), (400, 100), (500, 100), (600, 100)]
winner = None
reroll = False

# Run the game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if reroll:
                    winner, location = select_winner(names, locations)
                    if winner is not None:
                        reroll = False
                        print(f"{winner} wins!")
                        print(f"Selected location: {location}")
                else:
                    winner, location = select_winner(names, locations)
                    if winner is not None:
                        print(f"{winner} wins!")
                        print(f"Selected location: {location}")
    
    # Draw the background
    if reroll:
        screen.blit(reroll_image, (0, 0))
    else:
        screen.blit(bg_image, (0, 0))

    # Draw the names
    if winner is not None:
        font_surface = font.render(winner, True, (255, 255, 255))
        font_rect = font_surface.get_rect(center=location)
        screen.blit(font_surface, font_rect)

    # Update the display
    pygame.display.flip()

# Clean up
pygame.quit()
