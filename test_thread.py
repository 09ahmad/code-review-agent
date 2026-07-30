import threading 
 
def worker(): 
    from transformers import AutoModelForCausalLM 
    print("Worker thread import succeeded") 
 
threading.Thread(target=worker).start() 
