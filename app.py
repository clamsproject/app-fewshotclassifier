import argparse

# mostly likely you'll need these modules/classes
from clams import ClamsApp, Restifier, AppMetadata
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes


class YourApp(ClamsApp):

    def __init__(self, model_size="medium"):
        raise NotImplementedError

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._appmetadata
        raise NotImplementedError

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._annotate
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", action="store", default="5000", help="set port to listen"
    )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # more arguments as needed
    parsed_args = parser.parse_args()

    # create the app instance
    app = YourApp()

    http_app = Restifier(app, port=int(parsed_args.port)
    )
    if parsed_args.production:
        http_app.serve_production()
    else:
        http_app.run()