class AudioLike:
    """Mixin for Audio classes
    Define some basic operators on audio signals here.
    """
    @classmethod
    def open(cls, filename, *args, **kwargs):
        """open an image file, and read it as an 2d array
        
        Arguments:
            filename {str} -- file name of the image
        
        Returns:
            object of current cls
        """
        return cls.from_audio()


class AudioDouble(AudioLike, MultiEll1d):
    pass