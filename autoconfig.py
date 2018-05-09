
__all__ = ["AutoConfig"]

class _Section():
    def __init__(self):
        pass
    
class AutoConfig():
    def __init__(self, file_path):
        from strconv import convert
        from configparser import ConfigParser

        conf = ConfigParser()
        conf.read(file_path)

        for section in conf.sections():
            elem = _Section()
            setattr(AutoConfig, section, elem)
            for (k, v) in conf.items(section):
                setattr(elem, k, convert(v))