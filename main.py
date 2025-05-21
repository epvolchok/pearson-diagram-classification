#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import os
import logging
from libinteractive import InputManager, PathManager, ProcessingPipeline
from libservice import ServiceFuncs

def get_log_filename(log_dir='logs', log_name=None):
    """
    Determines a log file name.

    Parameters
    ----------
    log_dir : str
        Directory where log files are stored.
    log_name : str or None
        Custom log file name (e.g., 'log_experiment42.log').
        If None, a new log name will be generated as 'log_XX.log'.

    Returns
    -------
    str
        Full path to the log file.
    """
    os.makedirs(log_dir, exist_ok=True)

    if log_name:
        if not log_name.endswith('.log'):
            log_name += '.log'
        return os.path.join(log_dir, log_name)

    # Get existing log files
    existing_logs = [f for f in os.listdir(log_dir) if f.startswith('log_') and f.endswith('.log')]
    count = len(existing_logs)
    next_num = count + 1

    if count < 10:
        name = f'log_{next_num:02d}.log'  # log_01.log, log_02.log, ...
    else:
        name = f'log_{next_num}.log'      # log_11.log, log_12.log, ...

    return os.path.join(log_dir, name)

def setup_logging(log_file='pipeline.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            #logging.StreamHandler()
        ]
    )

def main():


    InputManager.welcome_message()

    
    input_imags, default_filename = PathManager.preparations()
    cwd = os.getcwd()
    default_info_path = os.path.join(cwd, 'data', 'SOLO_info_rswf.txt')
    message = f'Enter a path to the file with metadata (or press "Enter" to use default {default_info_path}): '
    info_path = PathManager.get_path(message, default_info_path)
    name_pattern = r'(solo_L2_rpw-tds-surv-(?:r|t)swf-e_\d+\w+)'

    features = ProcessingPipeline.get_features(input_imags, info_path, default_filename, name_pattern) #ResNet object

    print('Features are extracted/launched!')

    ProcessingPipeline.run_processing(features, default_filename)


if __name__ == '__main__':
    
    log_name = get_log_filename()
    setup_logging(log_name)
    
    main()


