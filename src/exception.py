import sys
from src.logger import logging

def error_message_details(error, error_details: sys):
   _, _, exc_tab = error_details.exc_info()   
   file_name = exc_tab.tb_frame.f_code.co_filename
   error_message = f"error accured in script: {file_name} at line {exc_tab.tb_lineno} error message: {str(error)}"

   return error_message

class CustomException(Exception):
   def __init__(self, error_message,error_details:sys):
      super().__init__(error_message)
      self.error_message = error_message_details(error_message, error_details)

   def __str__(self):
      return self.error_message


