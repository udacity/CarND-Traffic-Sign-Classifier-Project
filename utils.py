import matplotlib.pyplot as plt

def rescue_code(function):
    import inspect
    get_ipython().set_next_input("".join(inspect.getsourcelines(function)[0]))

def setup_figure(x_pics, y_pics, w, h, n=10, squeeze=True, sharex=True, sharey=True):
    return plt.subplots(y_pics, x_pics, squeeze=squeeze, sharex=sharex, sharey=sharey, figsize=(x_pics * n, y_pics * n))#h / w * n))        
