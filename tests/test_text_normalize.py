from voxcpm.utils.text_normalize import replace_blank


def test_replace_blank_keeps_interior_ascii_space():
    assert replace_blank("a b") == "a b"


def test_replace_blank_drops_edge_spaces():
    # A space is only kept between two ASCII word characters. A trailing space
    # used to raise IndexError (text[i + 1]) and a leading space was wrongly
    # kept (text[i - 1] wrapping to the last character); both are now dropped.
    assert replace_blank("hello ") == "hello"
    assert replace_blank(" hello") == "hello"
    assert replace_blank("a b ") == "a b"


def test_replace_blank_drops_space_adjacent_to_non_ascii():
    assert replace_blank("中 文") == "中文"
