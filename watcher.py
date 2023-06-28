import time
from watchdog.observers import Observer
import watchdog.events

class Watcher():
    def __init__(self, generations, individuals):
        self.observer = Observer()
        self.generations = generations
        self.generation = 0
        self.individuals = individuals
        self.individual = 0
        self.pending_generations = True
        self.pending_individuals = True

    def on_any_event(self, event):
        if event.is_directory:
            return None
        
        elif event.event_type == 'modified':
            file_full_name = event.src_path.split("/")
            file_name = file_full_name[-1].split(".")
            file_generation = file_name[0].split("_")

            self.generation = int(file_generation[1])

            with open(event.src_path, "r") as gen_file:
                lines = gen_file.readlines()
                if len(lines) <= 1:
                    self.individual = 0
                else:
                    last_line = lines[-1]
                    line_fields = last_line.split(";")
                    self.individual = int(line_fields[0])
                    self.individuals = int(line_fields[1])

 
    def run(self, watched_dir, gen_progress_bar, indi_progress_bar):
        self.event_handler = watchdog.events.PatternMatchingEventHandler(patterns=['*.csv'],
                                                             ignore_directories=True, case_sensitive=False)
        self.event_handler.on_any_event = self.on_any_event
        self.observer.schedule(self.event_handler, watched_dir, recursive = True)
        self.observer.start()

        try:
            while self.pending_generations or self.pending_individuals:
                if self.generation == self.generations:
                    self.pending_generations = False

                if self.individual == self.individuals:
                    self.pending_individuals = False
                else:
                    self.pending_individuals = True

                gen_label = 'Generation ' + str(self.generation)
                gen_progress_bar.progress(int(100*(self.generation)/(self.generations)), gen_label)

                indi_label = 'Individual ' + str(self.individual) + '/' + str(self.individuals)
                indi_progress_bar.progress(int(100*(self.individual)/(self.individuals)), indi_label)

                time.sleep(0.5)
        except:
            self.observer.stop()
