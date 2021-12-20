from time import time


class JobTimer:
    def __init__(self):
        self.time_1 = time()

    def __call__(self, job_done_fraction: float) -> None:
        time_2 = time()
        time_delta = time_2 - self.time_1
        speed = job_done_fraction / time_delta
        job_to_do_fraction = 1.0 - job_done_fraction
        time_to_end_estimate = job_to_do_fraction / speed
        time_to_end_estimate_hours = time_to_end_estimate / 3600.0
        txt = "time to end estimation: {x:.2f} hours"
        print(txt.format(x=time_to_end_estimate_hours))
