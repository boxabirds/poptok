"""Run generate_infinitetalk.py and report peak GPU memory and wall time at exit.
    python3 /opt/peak.py <generate_infinitetalk.py args>"""
import atexit, runpy, sys, time
import torch
t0 = time.time()
atexit.register(lambda: print(f"[peak] max_allocated={torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
                              f"max_reserved={torch.cuda.max_memory_reserved()/2**30:.2f} GiB wall={time.time()-t0:.0f}s",
                              flush=True))
sys.path.insert(0, "/opt/InfiniteTalk")
sys.argv = ["generate_infinitetalk.py"] + sys.argv[1:]
runpy.run_path("/opt/InfiniteTalk/generate_infinitetalk.py", run_name="__main__")
