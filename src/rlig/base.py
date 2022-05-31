class Base():
    
    def __init__(self, params, exceptions = []):
        super().__init__()
        self.__store__(params, exceptions)

    def __store__(self, params, exceptions = []):
        exceptions   += ['self', '__class__']
        self.__params = []
        for key, value in params.items():
            if key not in exceptions:
                setattr(self, key, value)
                self.__params.append(key)
        
        self.__name__ = self.__class__.__name__

    def __repr__(self):
        params = {key : getattr(self, key) for key in self.__params}
        params = [f'{key}={value}' for key, value in params.items()]
        return f'{self.__name__}({", ".join(params)})'
