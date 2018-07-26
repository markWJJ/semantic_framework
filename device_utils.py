import tensorflow as tf

DEFAULT_DEVICE_ID = -1
device_id = DEFAULT_DEVICE_ID

def next_device(use_cpu=True, num_gpu_cores=4, num_cpu_cores=8):
	global device_id
	
	if use_cpu:
		if device_id +1 < num_cpu_cores:
			device_id += 1
		device = "/cpu:%d" % device_id
	else:
		if device_id +1 < num_gpu_cores:
			device_id += 1
		device = "/gpu:%d" % device_id
	return device