import pygame.midi
import time

pygame.midi.init()

print (pygame.midi.get_default_output_id())
print(pygame.midi.get_default_input_id())
print( pygame.midi.get_device_info(0))

player = pygame.midi.Output(pygame.midi.get_default_output_id())

inPLayer = pygame.midi.Input(pygame.midi.get_default_input_id())
# player.write_short(148, 23 , 101)
player.write_short(148, 21, 101)
time.sleep(1)
# player.write_short(132, 23, 101)
player.write_short(132, 21, 101)
while True:
    if(inPLayer.poll()):
        events = inPLayer.read(1)
        if events[0][0][0] != 212:
            events[0][0][1] += 4
            events[0][0][2] += 4
        player.write_short(events[0][0][0],events[0][0][1],events[0][0][2])
        print(events)


pygame.midi.quit()