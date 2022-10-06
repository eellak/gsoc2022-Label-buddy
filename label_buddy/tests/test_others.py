from django.test import TestCase
import os


class administrative_tasks(TestCase):

    def administrative_tasks(self):
        os.environ.setdefault() 
        try: 
            from django.core.management import execute_from_command_line 
        except ImportError as exc: 
            raise ImportError( "Couldn't import Django. Are you sure it's installed and "
                    "available on your PYTHONPATH environment variable? Did you " 
                    "forget to activate a virtual environment?" 
                ) from exc 

        self.assertRaises(ImportError)