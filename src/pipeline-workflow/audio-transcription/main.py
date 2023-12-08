from cli import transcribe_file
import functions_framework


@functions_framework.http
def transcribe_http(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    print("request_json:", request_json)
    print("request_args:", request_args)


    filename = ""
    
    if request_args and "filename" in request_args:
        filename = request_args["filename"]

    print("Filename:",filename)
    results = transcribe_file(filename)
    print("Output:", results)

    return results