from collections import deque

memory = deque(maxlen=3)
memory.append(1)
memory.append(2)
memory.append(3)
print(memory)
memory.append(4)
memory.append(5)
print(memory)
memory.append(6)
