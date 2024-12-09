# Before using:
# sudo chmod 666 /dev/input/event0

# Also, sometimes you just have to reboot.

from evdev import InputDevice, categorize, ecodes
dev = InputDevice('/dev/input/event1')

print(dev)

for event in dev.read_loop():
    print(f"code: {event.code} type: {event.type} value: {event.value}")
