def track_called(func):
    def wrapper(self, *args, **kwargs):
        setattr(self, f"_{func.__name__}_called", True)
        return func(self, *args, **kwargs)
    return wrapper