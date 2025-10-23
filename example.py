from fpc_generator import *

def snake():
    cable_top = Cable(0.4, 0.4, 0.4, 1.6, 0.4, 0.8, 0.4)
    cable_bot = None
    sections = [
        Up(20),
        Curve(8),
        Right(12),
        Curve(3),
        Down(0),
        Curve(3),
        Left(12),
        Curve(2.5),
        Down(5)
    ]
    generate_cable("snake.kicad_mod", cable_top, cable_bot, sections)

if __name__ == "__main__":
    snake()
