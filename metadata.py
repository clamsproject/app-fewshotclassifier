"""
The purpose of this file is to define the metadata of the app with minimal imports. 

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    
    # first set up some basic information
    metadata = AppMetadata(
        name="Few Shot Classifier",
        description="This tool uses a vision model to classify video segments. Currenly supports \"chyron\" frame type.",
        app_license="MIT",
        identifier="fewshotclassifier",
        url="https://github.com/clamsproject/app-fewshotclassifier",
        analyzer_version="1.0",
        analyzer_license="MIT",
    )
    metadata.add_input(DocumentTypes.VideoDocument, required=True)
    metadata.add_output(AnnotationTypes.TimeFrame, frameType='string')
    
    metadata.add_parameter(name='timeUnit', type='string', default='frames', 
                           choices=['frames', 'milliseconds'],
                           description='Unit for output timeframe')
    metadata.add_parameter(name='sampleRatio', type='integer', default='30',
                           description='Frequency to sample frames.')
    metadata.add_parameter(name='minFrameCount', type='integer', default='60',
                           description='Minimum number of frames required for a timeframe to be included in the '
                                       'output with a minimum value of 1')
    metadata.add_parameter(name='threshold', type='number', default='.8',
                           description='Threshold from 0-1, lower accepts more potential labels.')
    metadata.add_parameter(name='finetunedFrameType', type='string', default='chyron', 
                           description='Name of fine-tuned model to use. All pre-installed models are named after the frame type they were fine-tuned for.\nIf an empty value is passed, the app will look for fewshots.csv file in the same directory as the app.py and create a new fine-tuned model based on the examples in that file.\nAt the moment, a model fine-tuned on "chyron" frame type is shipped as pre-installed.')
                           
                           

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
