import threading

def writer1(event_for_wait, event_for_set):
    while True:
        x = input('X: ')
        return x

def writera(x, event_for_wait, event_for_set):
    while True:
        event_for_wait.wait()  # wait for event
        event_for_wait.clear()  # clean event for future
        print(x)
        event_for_set.set()  # set event for neighbor thread

# init events
e1 = threading.Event()
e2 = threading.Event()
e3 = threading.Event()

# init threads
t1 = threading.Thread(target=writer1, args=(e1, e2))
t2 = threading.Thread(target=writera, args=(writer1(), e2, e3))
t3 = threading.Thread(target=writer1, args=(2, e3, e1))

# start threads
t1.start()
t2.start()
t3.start()

e1.set() # initiate the first event

# join threads to the main thread
t1.join()
t2.join()
t3.join()