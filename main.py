#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0


import matplotlib.pyplot as plt
import os
import re

from libpreprocessing import FeaturesPreprocessing
from libclustering import Clustering
from libfeatures import ResNetFeatures
from libservice import ServiceFuncs
from libinteractive import InteractiveMode

def main():


    InteractiveMode.welcome_message()

    
    input_imags, default_filename = InteractiveMode.preparations()
    default_info_path = './data/SOLO_info_rswf.txt'
    message = f'Enter a path to the file with metadata (or enter to use default {default_info_path}): '
    info_path = InteractiveMode.get_path(message, default_info_path)
    name_pattern = r'(solo_L2_rpw-tds-surv-(?:r|t)swf-e_\d+\w+)'

    features = InteractiveMode.get_features(input_imags, info_path, default_filename, name_pattern) #ResNet object

    print('Features are extracted/launched!')

    InteractiveMode.run_processing(features, default_filename)


if __name__ == '__main__':
    main()


