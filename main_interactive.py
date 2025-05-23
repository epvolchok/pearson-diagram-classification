#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import os
import logging
from minteractive import*
from libservice import ServiceFuncs, Logg

logger = logging.getLogger(__name__)

def main():


    InputManager.welcome_message()

    
    input_imags, default_filename = PathManager.preparations()
    cwd = os.getcwd()
    default_info_path = os.path.join(cwd, 'data', 'SOLO_info_rswf.txt')
    message = f'Enter a path to the file with metadata (or press "Enter" to use default {default_info_path}): '
    info_path = PathManager.get_path(message, default_info_path)
    name_pattern = r'(solo_L2_rpw-tds-surv-(?:r|t)swf-e_\d+\w+)'

    features = FeatureManager.get_features(input_imags, info_path, default_filename, name_pattern) #ResNet object

    print('Features are extracted/launched!')

    ProcessingPipeline.run_processing(features, default_filename)


if __name__ == '__main__':
    
    log_name = Logg.get_log_filename()
    Logg.setup_logging(log_name)

    main()
