def schematize(object, cache):
    if hasattr(object, "cache_id"):
        cache[str(object.cache_id)] = object
    if hasattr(object, "sm_schematize"):
        return object.sm_schematize(cache)
    if isinstance(object, str):
        return {"schema": "text", "value": object}
    if object is None:
        return {"schema": "nothing"}
    if isinstance(object, dict):
        if "schema" in object:
            return object
    

    return {"schema": "other", "description": f"{object}"}
