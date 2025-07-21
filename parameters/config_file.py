import argparse
import os 
import sys 

# Append project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from parameters.parameters import parameters_details


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Backtesting Framework Configuration')
        self.setup_arguments()

    def setup_arguments(self):
        # Dynamically add arguments from the parameters_details dictionary
        for param, details in parameters_details.items():
            # Use the provided type, default, and help from the dictionary
            self.parser.add_argument(
                f'--{param}',
                type=details['type'],
                default=details['default'],
                help=details['help'],
                choices=details.get('choices')  # This will be None if not provided
            )

    def parse_args(self, args=None):
        # Parse the arguments provided via command line or passed as a list
        return self.parser.parse_args(args)


