from pathlib import Path

from sagemaker_shim.vendor.werkzeug.security import safe_join


def test_safe_join():
    assert safe_join("foo", "bar/baz") == Path("foo/bar/baz")
    assert safe_join("foo", "../bar/baz") is None


def test_safe_join_os_sep():
    import sagemaker_shim.vendor.werkzeug.security as sec

    prev_value = sec._os_alt_seps
    sec._os_alt_seps = "*"
    assert safe_join("foo", "bar/baz*") is None
    sec._os_alt_steps = prev_value


def test_safe_join_empty_trusted():
    assert safe_join("", "c:test.txt") == Path("c:test.txt")


def test_empty_filename():
    assert safe_join("foo", "") == Path("foo")
