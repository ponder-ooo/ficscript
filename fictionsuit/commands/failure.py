class CommandFailure(str):
    def __new__(cls, *args, **kwargs):
        return str.__new__(cls, *args, **kwargs)

    def sm_schematize(self, cache):
        return {"schema": "failure", "explanation": self}
