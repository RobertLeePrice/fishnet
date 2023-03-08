valid_extensions = ["jpg", "png", "tiff", "tif"]

# InvalidFileTypeException
class InvalidFileTypeError(Exception):
    """Raised when the input file has an invalid format"""
    def __init__(self):
        msg = "The image entered is flagged as a valid type but is currently "
        msg += "unsupported. This is an error with the script itself and not "
        msg += "the user unless the user changed the script."
        super().__init__(msg)

class FileNotFoundError(Exception):
    """Raised when the input file is not found"""


class InvalidFileTypeException(Exception):
   def __init__(self, valid_extensions):
      msg = f"This script only accepts images of the following type {', '.join(valid_extensions)}"
      super().__init__(msg)

class UnsupportedFileTypeException(Exception):
   def __init__(self):
      msg = "The image entered is flagged as a valid type but is currently "
      msg += "unsupported. This is an error with the script itself and not "
      msg += "the user unless the user changed the script."
      super().__init__(msg)

class MissingInputException(Exception):
   def __init__(self):
      msg = "This script expects exactly 1 input but nothing was input. "
      msg += "Rerun the script with the image file you want to process."
      super().__init__(msg)

class ExcessiveInputException(Exception):
   def __init__(self):
      msg = "This script expects exactly 1 input but more than one was "
      msg += "input with the script. Rerun the script with exactly one "
      msg += "file path"
      super().__init__(msg)