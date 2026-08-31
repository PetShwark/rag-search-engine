def rating_text_to_float(rating_text: str) -> float:
    """
    If the rating text cannot be turned into a float, just return 0.0.  With the free models, we often get
    "User Safety: safe" returned from the model when things go wrong.  This will keep us from crashing, but
    will mess up our rankings.  We don't care right now.
    """
    result = 0.0
    try:
        result = float(rating_text)
    except:
        pass
    return result