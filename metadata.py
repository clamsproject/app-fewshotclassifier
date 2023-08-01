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
        description="This tool uses a vision model to classify video segments by comparing them to examples",
        app_license="MIT",
        identifier="fewshotclassifier",
        url="https://github.com/clamsproject/app-fewshotclassifier",
        analyzer_version="1.0",
        analyzer_license="MIT",
    )
    metadata.add_input(DocumentTypes.VideoDocument, required=True)
    metadata.add_output(AnnotationTypes.TimeFrame, frameType='string')
    
    metadata.add_parameter(name='timeUnit', description='Unit for output timeframe',
                           type='string', default='milliseconds', choices=['frames', 'milliseconds'])
    metadata.add_parameter(name='sampleRatio', description='Frequency to sample frames.',
                           type='integer', default='10')
    metadata.add_parameter(name='minFrameCount', type='integer', default='10',
                           description='Minimum number of frames required for a timeframe to be included in the output'
                                       'with a minimum value of 1')
    metadata.add_parameter(name='threshold', type='number', default='.9',
                           description='Threshold from 0-1, lower accepts more potential labels.')
    metadata.add_parameter(name='cutoffMins', type='integer', description='Maximum number of minutes to process')

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
