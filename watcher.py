import time
from watchdog.observers import Observer
import watchdog.events

class Watcher():
    def __init__(self, iterations, label):
        self.observer = Observer()
        self.iterations = iterations
        self.label = label
        self.iteration = 0

    def on_any_event(self, event):
        if event.is_directory:
            return None
 
        elif event.event_type == 'created':
            self.iteration += 1

 
    def run(self, watched_dir, progress_bar):
        self.event_handler = watchdog.events.PatternMatchingEventHandler(patterns=['*.csv'],
                                                             ignore_directories=True, case_sensitive=False)
        self.event_handler.on_any_event = self.on_any_event
        self.observer.schedule(self.event_handler, watched_dir, recursive = True)
        self.observer.start()

        try:
            while self.iteration <= self.iterations:
                it = self.iteration+1
                label = self.label + ' ' + str(it)
                progress_bar.progress(int(100*(self.iteration+1)/self.iterations), label)
                time.sleep(0.1)
        except:
            self.observer.stop()
        
        self.observer.join()
